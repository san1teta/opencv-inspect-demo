"""放位检查：拍一张，告诉你硬币离「标定纸心」偏了多少毫米、该往哪边挪。

ppm 是标定图四条边的均值，只在纸心那一点严格成立，偏离越远比例尺误差越大
（本装置实测上下方向 0.74% 峰峰，纸心到纸边 ±0.093mm）。
关键：纸心 **不等于画面中心** —— 本装置差 69mm，凭感觉对准镜头正下方反而会偏。

三项必须一起验。只验位置的版本放过了一张废片：硬币量出 22.17mm（1元 标称 25.0），
偏移 9.1mm 于是报「位置合格」—— 可 ppm 已经掉了 10%，位置再准也测不出东西。
比例尺和清晰度这两关就是为堵这个洞加的。

用法：python tools/check_placement.py <一张放着硬币的试拍图> [--coin 1元]
退出码 0=三项全过，1=有不合格项（可直接挂进批处理/CI）
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

# 清晰度闸门：外缘 10~90% 过渡宽度。锐利边实测 1~2px；糊到 3px 以上时边缘定位
# 抖动就会把 0.25mm 允差吃光。同一张糊片（8.5px）的圆形度从完美圆的 0.99 掉到 0.904，
# 已经贴着 min_circularity=0.9 的门槛 —— 再糊一点就整批判不出。
BLUR_GUARD_PX = 3.0


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
    if top < bot:                      # 暗币亮底：翻过来按同一条下降沿处理
        prof, top, bot = 255.0 - prof, 255.0 - bot, 255.0 - top
    if top - bot < 12:                 # 对比度不够，边缘根本没成形，不拿它下结论
        return float('nan')
    g = np.gradient(prof)
    j = int(np.argmin(g))              # 最陡的下落点 = 边缘
    if j <= 0 or j >= n - 1:
        return float('nan')

    def cross(lev: float, direction: int) -> float:
        i = j
        while 0 <= i + direction < n:
            a, b = prof[i], prof[i + direction]
            if (a >= lev) != (b >= lev) and a != b:
                return r0 + (i + direction * (lev - a) / (b - a)) * step
            i += direction
        return float('nan')

    r_hi = cross(bot + 0.9 * (top - bot), -1)
    r_lo = cross(bot + 0.1 * (top - bot), +1)
    if math.isnan(r_hi) or math.isnan(r_lo):
        return float('nan')
    return float(r_lo - r_hi)


def verdict(dist_mm: float, dev_mm: float | None, blur_px: float,
            tol_mm: float = PLACE_TOL_MM) -> tuple[str, list[str]]:
    """三项判据汇总成 (代号, 待修事项)。任一不过就不给 OK。

    顺序按「先修哪个才不会白拍」排：比例尺 > 清晰度 > 位置。
    比例尺错了整批数据全废；清晰度错了重拍就能救；位置偏了挪一下就行。
    """
    bad: list[str] = []
    if dev_mm is not None and abs(dev_mm) > SCALE_GUARD_MM:
        bad.append(f'比例尺：直径离标称 {dev_mm:+.2f}mm，ppm 已失效，重拍标定图并 --write')
    if not math.isnan(blur_px) and blur_px > BLUR_GUARD_PX:
        bad.append(f'清晰度：外缘 10-90% 过渡 {blur_px:.1f}px > {BLUR_GUARD_PX:g}px，'
                   f'对上硬币点屏幕锁定对焦/AE，或加光缩短快门')
    if dist_mm > tol_mm:
        bad.append(f'位置：偏离纸心 {dist_mm:.1f}mm > {tol_mm:g}mm')
    return ('OK' if not bad else 'NG'), bad



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

    gray = preprocess.gray(img)
    edges = preprocess.canny(preprocess.gaussian_blur(gray, config_setting.gaussian_kernel),
                             config_setting.canny_low, config_setting.canny_high)
    coins = [detector.calculate_features(c)
             for c in detector.find_contours(edges, config_setting.min_area)]
    coins = [f for f in coins if f['circularity'] >= config_setting.min_circularity]
    if not coins:
        sys.exit("没找到可信的圆形目标：硬币没入画、被遮挡，或者对比度不够")

    coin = args.coin or guess_coin(args.image)
    spec = config_setting.coin_specs.get(coin)

    scale = max(0.5, w / 1200)
    thick = max(1, int(round(scale * 2)))
    vis = img.copy()
    cv2.drawMarker(vis, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, int(50 * scale), thick)
    print(f"标定纸心 ({tx:.0f}, {ty:.0f}) | 画面中心 ({w // 2}, {h // 2}) —— 纸心在画面中心"
          f"{'上' if ty < h / 2 else '下'} {abs(ty - h / 2) / ppm:.0f}mm，别按画面中心放")
    print(f"比例尺闸门：面额 {coin or '未识别（跳过直径校验）'}"
          + (f" 标称 {spec['diameter']}mm，容差 ±{SCALE_GUARD_MM}mm" if spec else ''))

    failed = 0
    for i, f in enumerate(sorted(coins, key=lambda x: -x['area']), 1):
        cx, cy = f['center']
        dx, dy, parts = nudge(tx, ty, cx, cy, ppm)
        dist = (dx * dx + dy * dy) ** 0.5 / ppm
        dia = f['diameter_pixels'] / ppm
        dev = round(dia - spec['diameter'], 3) if spec else None
        blur = edge_blur_px(gray, cx, cy, f['diameter_pixels'] / 2)
        tag, bad = verdict(dist, dev, blur)
        failed += bool(bad)
        col = (0, 200, 0) if tag == 'OK' else (0, 0, 255)
        cv2.circle(vis, (cx, cy), int(f['diameter_pixels'] / 2), col, thick)
        cv2.line(vis, (cx, cy), (int(tx), int(ty)), (0, 0, 255), thick)
        b = '' if math.isnan(blur) else f" 缘宽{blur:.1f}px"
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
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
