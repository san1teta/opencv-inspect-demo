"""放位检查：拍一张，告诉你硬币离「标定纸心」偏了多少毫米、该往哪边挪。

ppm 是标定图四条边的均值，只在纸心那一点严格成立，偏离越远比例尺误差越大
（本装置实测上下方向 0.74% 峰峰，纸心到纸边 ±0.093mm）。
关键：纸心 **不等于画面中心** —— 本装置差 69mm，凭感觉对准镜头正下方反而会偏。

三项必须一起验。只验位置的版本放过了一张废片：硬币量出 22.17mm（1元 标称 25.0），
偏移 9.1mm 于是报「位置合格」—— 可 ppm 已经掉了 10%，位置再准也测不出东西。
比例尺和清晰度这两关就是为堵这个洞加的。

还有第四关「照片新鲜度」，它堵的是另一类错：闸门只能保证「图与 config 自洽」，
不能保证「config 对应当时的装置」。一张旧图配一次错的新标定，三关照样全绿。

用法：python tools/check_placement.py <一张放着硬币的试拍图> [--coin 1元]
退出码 0=全部通过，1=有不合格项（可直接挂进批处理/CI）
"""
import argparse
import io
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from calib_guard import photo_vs_calibration  # noqa: E402
import config_setting  # noqa: E402
import detector  # noqa: E402
import image_io  # noqa: E402
import preprocess  # noqa: E402

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')

# 本装置实测比例尺梯度 0.74% 峰峰，待测物偏离纸心 ±57mm 以内该项误差 <0.05mm，取 50 留余量
PLACE_TOL_MM = 50.0

# 比例尺闸门：真币磨损 <0.5mm、公差 0.25mm，量得值离标称 2mm 以上绝不可能是币的问题，
# 只能是 ppm 和拍摄时的实际距离脱了节。实测踩过的坑：标定后手机被碰离 28→31cm，
# 1元 量成 22.17mm（-11%），而当时工具只看偏移，照样报「合格」。
SCALE_GUARD_MM = 2.0

# 清晰度闸门：外缘 10~90% 过渡宽度。**必须按毫米定，不能按像素定。**
# 原先写死 3px：今天重标定 ppm 从 9.9749 掉到 8.6620，同一个 3px 悄悄从 0.30mm
# 变成 0.35mm —— 代码一个字没改，判据却松了 15%，而且不会有任何报错。
# 这种「数值没变但含义变了」的坑本项目已经反复踩，所以把物理量留在毫米域，
# 像素只在测量时用一次。锐利边实测 1~2px（旧 ppm 下），0.30mm 对应原来的 3px。
BLUR_GUARD_MM = 0.30


def guess_coin(path: str) -> str:
    """从目录名/文件名里认面额，认不出返回 ''（则跳过比例尺闸门）。

    目录约定就是 shots/coins/<面额>/xxx.jpg，所以取路径全串匹配 coin_specs 的键。
    长键优先，免得「1元」被「1元角」之类误吞。
    """
    name = os.path.abspath(path)
    for key in sorted(config_setting.coin_specs, key=len, reverse=True):
        if key in name:
            return key
    return ''


def edge_blur_px(gray: np.ndarray, cx: float, cy: float, r_edge: float) -> float:
    """沿半径做环形平均，返回外缘 10~90% 过渡宽度（px）。越大越糊；测不出返回 nan。

    为什么不用 Laplacian 方差之类的全图清晰度：全图清晰度会被背景纹理和 JPEG 噪声
    主导，币边缘糊不糊得看边缘本身。环形平均同时把 720 个方向的噪声摊平，
    分辨率能到 0.5px 一档，足够卡 3px 这条线。
    """
    h, w = gray.shape
    band = max(8.0, r_edge * 0.2)
    r0, step, n = max(1.0, r_edge - band), 0.5, int(2 * band / 0.5)
    if n < 8:
        return float('nan')
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    prof = np.empty(n)
    for i in range(n):
        r = r0 + i * step
        xs = np.clip((cx + r * ct).astype(int), 0, w - 1)
        ys = np.clip((cy + r * st).astype(int), 0, h - 1)
        prof[i] = gray[ys, xs].mean()
    k = max(1, n // 8)
    top, bot = float(prof[:k].mean()), float(prof[-k:].mean())
    if top < bot:                      # 暗币亮底：整条剖面翻过来，按同一条下降沿处理
        # 注意 top/bot 都只取反、**不能对调**：prof[i] <- 255-prof[i] 之后，
        # 内侧窗口均值仍是 255-top、外侧仍是 255-bot，对调会让 top 依旧小于 bot，
        # 下一步 `top - bot < 12` 必然成立 → 暗币亮底整条分支从来没工作过（新测试抓到）。
        prof, top, bot = 255.0 - prof, 255.0 - top, 255.0 - bot
    if top - bot < 12:                 # 对比度不够，边缘根本没成形，不拿它下结论
        return float('nan')
    g = np.gradient(prof)
    j = int(np.argmin(g))              # 最陡的下落点 = 边缘
    if j <= 0 or j >= n - 1:
        return float('nan')

    def cross(lev: float) -> float:
        """在所有跨越 lev 的相邻采样里，取离最陡点 j 最近的那个。

        不能只朝一个方向找。真实硬币剖面不是对称台阶：币缘内侧有一圈亮带
        （齿边/浮雕），而 `top` 取的是最内侧窗口均值，会被这圈亮带骗低。
        a01.jpg 实测 top=77.8 而 r=93.5 处高达 105.8，90% 电平算出来只有 74.3
        → 整条左段全在电平之上，交点其实落在 j 的**右侧**（i=38.96，[39,40] 段）。
        原来写死 `direction=-1` 往左找，找不到就返回 nan，
        **清晰度闸门于是在第一张真照片上整条静默不执行**（判据见 verdict：nan 不算 NG）。
        """
        best, best_d = float('nan'), float('inf')
        for i in range(n - 1):
            a, b = float(prof[i]), float(prof[i + 1])
            if a == b or (a >= lev) == (b >= lev):
                continue
            d = abs(i + (lev - a) / (b - a) - j)
            if d < best_d:
                best, best_d = i + (lev - a) / (b - a), d
        return r0 + best * step if math.isfinite(best) else float('nan')

    r_hi = cross(bot + 0.9 * (top - bot))
    r_lo = cross(bot + 0.1 * (top - bot))
    # r_lo <= r_hi 说明两个交点取自不同的边沿（剖面被浮雕切成多段），
    # 这时候算出来的「宽度」是假的，宁可返回 nan 让闸门明说自己没测到。
    if math.isnan(r_hi) or math.isnan(r_lo) or r_lo <= r_hi:
        return float('nan')
    return float(r_lo - r_hi)


def verdict(dist_mm: float, dev_mm: float | None, blur_mm: float,
            tol_mm: float = PLACE_TOL_MM) -> tuple[str, list[str]]:
    """三项判据汇总成 (代号, 待修事项)。任一不过就不给 OK。三个量全在毫米域。

    顺序按「先修哪个才不会白拍」排：比例尺 > 清晰度 > 位置。
    比例尺错了整批数据全废；清晰度错了重拍就能救；位置偏了挪一下就行。
    """
    bad: list[str] = []
    if dev_mm is not None and abs(dev_mm) > SCALE_GUARD_MM:
        bad.append(f'比例尺：直径离标称 {dev_mm:+.2f}mm，ppm 已失效，重拍标定图并 --write')
    if not math.isnan(blur_mm) and blur_mm > BLUR_GUARD_MM:
        bad.append(f'清晰度：外缘 10-90% 过渡 {blur_mm:.3f}mm > {BLUR_GUARD_MM:g}mm，'
                   f'改手动对焦并放大核对纸缘/币缘锐度，或加光缩短快门')
    if dist_mm > tol_mm:
        bad.append(f'位置：偏离纸心 {dist_mm:.1f}mm > {tol_mm:g}mm')
    return ('OK' if not bad else 'NG'), bad


def stale_note(path: str, calib_stamp: str) -> str:
    """照片早于标定时刻 => 它是旧装置拍的，不能拿来验收当前 ppm。没问题返回空串。

    判据本身在 `coin_detect/calib_guard.py` —— 批量流水线也用它，理由见那个模块：
    这道闸门原先只长在验收工具上，旧照片留在样本目录里会被新 ppm 静默量掉。
    为什么非拦不可：昨天那张 test.jpg 实际比例尺 8.844，今天重标定是 8.6620，
    差 2.1% —— 25mm 上是 0.53mm、两倍于允差，可 `SCALE_GUARD_MM=2.0` 那道闸门
    根本看不见它，三关全绿通过。**用错一张过期照片，工具会点头。**
    """
    times = photo_vs_calibration(path, calib_stamp)
    if times is None:
        return ''
    shot, calib = times
    return (f'照片比标定旧 {int((calib - shot).total_seconds() // 60)} 分钟'
            f'（拍摄 {shot:%m-%d %H:%M} < 标定 {calib:%m-%d %H:%M}）'
            f'—— 是重标定之前的装置拍的，不能用来验收当前 ppm，重拍')


def explain_no_coin(img) -> list[str]:
    """检不到币时，把画面里最大的轮廓摊开来说话，而不是甩一句「没找到」。

    为什么非做不可：真实原因九成是 **A4 参照纸没撤**。detector.find_contours
    用 RETR_EXTERNAL 只取最外层轮廓，硬币压在纸上时币属于**嵌套轮廓，会被整个
    丢掉**，只剩一张纸 —— 于是报「没找到可信的圆形目标」，而现场人完全看不出
    自己错在哪。实测 a01.jpg 就是这么废的。
    """
    g = preprocess.gray(img)
    edges = cv2.Canny(preprocess.gaussian_blur(g, config_setting.gaussian_kernel),
                      config_setting.canny_low, config_setting.canny_high)
    contours = detector.find_contours(edges, config_setting.min_area)
    if not contours:
        return ['一个轮廓都没有：检查曝光/对比度，或镜头是否被挡']
    f = max((detector.calculate_features(c) for c in contours),
            key=lambda x: x['area'])
    dia = f['diameter_pixels'] / config_setting.ppm
    cx, cy = f['center']
    out = [f'画面最大轮廓：等效直径 {dia:.0f}mm，圆形度 {f["circularity"]:.3f}，'
           f'中心 ({cx:.0f}, {cy:.0f})']
    if dia > 60:
        out.append(f'  {dia:.0f}mm 远超任何硬币（最大 25mm），这不是币')
        tx, ty = config_setting.calib_center
        if abs(cx - tx) < 200 and abs(cy - ty) < 200:
            out.append('  中心又与标定纸心重合 => 就是那张 A4 参照纸。'
                       '纸只用于标定，量硬币前必须撤掉：币压在纸上会被当嵌套轮廓丢掉')
    else:
        out.append('  最大轮廓比硬币还小，说明币没入画或被遮挡；'
                   '若币在画面里但圆形度偏低，多半是虚焦或币面反光')
    return out


def nudge(tx: float, ty: float, cx: float, cy: float, ppm: float,
          min_mm: float = 2.0) -> tuple[float, float, list[tuple[str, float]]]:
    """把「硬币当前圆心」挪到「标定纸心」该往哪边走，返回 (dx, dy, 建议列表)。

    图像 y 轴向下，所以 dy>0（硬币在目标下方）要往「上」挪 —— 方向极易写反，
    写反了还会照常工作、只是把人往错处支使，所以单独抽成纯函数配测试。
    """
    dx, dy = cx - tx, cy - ty
    parts = []
    if abs(dy) / ppm > min_mm:
        parts.append(('下' if dy < 0 else '上', abs(dy) / ppm))
    if abs(dx) / ppm > min_mm:
        parts.append(('右' if dx < 0 else '左', abs(dx) / ppm))
    return dx, dy, parts


def pick_coins(feats: list[dict], spec: dict | None, ppm: float) -> list[dict]:
    """从候选轮廓里挑「像是这枚面额的币」。抽成纯函数是为了能直接测。

    有面额 → 用与 pipeline 同一个 `diameter_window`。圆度低（变形、被贴胶带）也**必须留下**：
    那是质量结论，要让它活到报表里被如实判不合格，而不是在这里被当背景悄悄删掉。
    这正是昨天 51 个假币那条坑的根因 —— 工具一条路、流水线一条路各长一套闸门。

    没面额 → 没有标称直径可依，尺寸窗口无从构造，退回圆度筛。这是**降级**不是等价方案，
    调用方会把「跳过直径校验」明说出来，不假装这道闸门有效。
    """
    if spec:
        lo_px, hi_px = detector.diameter_window(spec['diameter'], ppm,
                                                config_setting.diameter_gate_mm)
        return [f for f in feats if lo_px <= f['diameter_pixels'] <= hi_px]
    return [f for f in feats if f['circularity'] >= config_setting.min_circularity]


def main() -> None:
    ap = argparse.ArgumentParser(description="硬币试拍验收：位置 / 比例尺 / 清晰度")
    ap.add_argument("image")
    ap.add_argument("--coin", default="", help="面额，默认从路径里的 1元/5角/1角 自动认")
    ap.add_argument("--out", default="", help="标注图路径，默认同目录 place_*.jpg")
    args = ap.parse_args()

    tx, ty = config_setting.calib_center
    if (tx, ty) == (0.0, 0.0):
        sys.exit("config 里 calib_center 还是 0，先跑 tools/measure_ppm.py <标定图> --write")
    ppm = config_setting.ppm

    img = image_io.read_image(args.image)
    h, w = img.shape[:2]
    if (w, h) != tuple(config_setting.calib_image_size):
        print(f"警告：本图 {w}x{h} 与标定图 {tuple(config_setting.calib_image_size)} 不一致，ppm 不适用")
    stale = stale_note(args.image, getattr(config_setting, 'calib_stamp', ''))
    if stale:
        print(f"[NG] {stale}")

    gray = preprocess.gray(img)
    edges = preprocess.canny(preprocess.gaussian_blur(gray, config_setting.gaussian_kernel),
                             config_setting.canny_low, config_setting.canny_high)
    coins = [detector.calculate_features(c)
             for c in detector.find_contours(edges, config_setting.min_area)]
    coin = args.coin or guess_coin(args.image)
    spec = config_setting.coin_specs.get(coin)
    coins = pick_coins(coins, spec, ppm)
    if not coins:
        print('[NG] 没找到可信的圆形目标，先看清为什么：')
        for line in explain_no_coin(img):
            print('  ' + line)
        sys.exit(1)

    scale = max(0.5, w / 1200)
    thick = max(1, int(round(scale * 2)))
    vis = img.copy()
    cv2.drawMarker(vis, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, int(50 * scale), thick)
    print(f"标定纸心 ({tx:.0f}, {ty:.0f}) | 画面中心 ({w // 2}, {h // 2}) —— 纸心在画面中心"
          f"{'上' if ty < h / 2 else '下'} {abs(ty - h / 2) / ppm:.0f}mm，别按画面中心放")
    print(f"比例尺闸门：面额 {coin or '未识别（跳过直径校验，候选退回圆度筛）'}"
          + (f" 标称 {spec['diameter']}mm，容差 ±{SCALE_GUARD_MM}mm" if spec else ''))

    failed = 0
    for i, f in enumerate(sorted(coins, key=lambda x: -x['area']), 1):
        cx, cy = f['center']
        dx, dy, parts = nudge(tx, ty, cx, cy, ppm)
        dist = (dx * dx + dy * dy) ** 0.5 / ppm
        dia = f['diameter_pixels'] / ppm
        dev = round(dia - spec['diameter'], 3) if spec else None
        blur_px = edge_blur_px(gray, cx, cy, f['diameter_pixels'] / 2)
        blur_mm = blur_px / ppm
        tag, bad = verdict(dist, dev, blur_mm)
        failed += bool(bad)
        col = (0, 200, 0) if tag == 'OK' else (0, 0, 255)
        cv2.circle(vis, (cx, cy), int(f['diameter_pixels'] / 2), col, thick)
        cv2.line(vis, (cx, cy), (int(tx), int(ty)), (0, 0, 255), thick)
        b = '' if math.isnan(blur_px) else f" 缘宽{blur_mm:.3f}mm({blur_px:.1f}px)"
        print(f"  目标{i} 圆心({cx},{cy}) 直径 {dia:.3f}mm{b}  偏离纸心 {dist:.1f}mm [{tag}]")
        for msg in bad:
            print(f"        [x] {msg}")
        if dist > PLACE_TOL_MM and parts:
            print(f"        [->] 位置怎么挪：把硬币往画面 "
                  f"{' + '.join(f'{d} {mm:.0f}mm' for d, mm in parts)}")

    stem = re.sub(r'[^0-9A-Za-z_.-]', '', os.path.splitext(os.path.basename(args.image))[0])[-24:]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                   f"place_{stem or 'coin'}.jpg")
    if not image_io.write_image(out, vis):
        raise RuntimeError(f"标注图写盘失败: {out}")
    print(f"标注图（红十字=标定纸心，绿圆=通过，红圆=不合格）: {out}")
    if failed:
        print(f"{failed} 个目标未通过 —— 先修「比例尺」再修「清晰度」，最后才管位置，顺序反了就白拍")
    if stale:
        print('照片早于标定，这一张不作数 —— 重拍后再跑（旧图三关全绿也不能信，见 stale_note）')
    sys.exit(1 if (failed or stale) else 0)


if __name__ == "__main__":
    main()
