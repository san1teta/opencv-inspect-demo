"""放位检查：拍一张，告诉你硬币离「标定纸心」偏了多少毫米、该往哪边挪。

ppm 是标定图四条边的均值，只在纸心那一点严格成立，偏离越远比例尺误差越大
（本装置实测上下方向 0.74% 峰峰，纸心到纸边 ±0.093mm）。
关键：纸心 **不等于画面中心** —— 本装置差 69mm，凭感觉对准镜头正下方反而会偏。

用法：python tools/check_placement.py <一张放着硬币的试拍图>
"""
import argparse
import io
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import config_setting  # noqa: E402
import detector  # noqa: E402
import image_io  # noqa: E402
import preprocess  # noqa: E402

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')

# 本装置实测比例尺梯度 0.74% 峰峰，待测物偏离纸心 ±57mm 以内该项误差 <0.05mm，取 50 留余量
PLACE_TOL_MM = 50.0


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
    ap = argparse.ArgumentParser(description="硬币放位检查")
    ap.add_argument("image")
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

    scale = max(0.5, w / 1200)
    thick = max(1, int(round(scale * 2)))
    vis = img.copy()
    cv2.drawMarker(vis, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, int(50 * scale), thick)
    print(f"标定纸心 ({tx:.0f}, {ty:.0f}) | 画面中心 ({w // 2}, {h // 2}) —— 纸心在画面中心"
          f"{'上' if ty < h / 2 else '下'} {abs(ty - h / 2) / ppm:.0f}mm，别按画面中心放")

    for i, f in enumerate(sorted(coins, key=lambda x: -x['area']), 1):
        cx, cy = f['center']
        dx, dy, parts = nudge(tx, ty, cx, cy, ppm)
        dist = (dx * dx + dy * dy) ** 0.5 / ppm
        cv2.circle(vis, (cx, cy), int(f['diameter_pixels'] / 2), (0, 255, 0), thick)
        cv2.line(vis, (cx, cy), (int(tx), int(ty)), (0, 0, 255), thick)
        if dist <= PLACE_TOL_MM:
            advice, tag = '位置合格', 'OK'
        else:
            way = ' + '.join(f'{d} {mm:.0f}mm' for d, mm in parts)
            advice, tag = f'把硬币往画面 {way} 挪', '挪'
        print(f"  目标{i} 圆心({cx},{cy}) 直径 {f['diameter_pixels'] / ppm:.3f}mm  "
              f"偏离纸心 {dist:.1f}mm [{tag}] {advice}")

    stem = re.sub(r'[^0-9A-Za-z_.-]', '', os.path.splitext(os.path.basename(args.image))[0])[-24:]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                   f"place_{stem or 'coin'}.jpg")
    if not image_io.write_image(out, vis):
        raise RuntimeError(f"标注图写盘失败: {out}")
    print(f"标注图（红十字=标定纸心，绿圆=检出的币，红线=偏移）: {out}")


if __name__ == "__main__":
    main()
