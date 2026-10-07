"""把程序"看到的"东西画出来：Otsu 二值化、选中的轮廓、外接矩形，缩图存盘。

标定不合格时先看这张图，别猜。
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import calibration  # noqa: E402


def read_image(path: str) -> np.ndarray:
    """中文路径 cv2.imread 读不了（会按 ANSI 解码），必须用 imdecode 兜底。"""
    img = cv2.imread(path)
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 标定诊断")
    ap.add_argument("image")
    ap.add_argument("--long", type=float, default=calibration.A4_LONG_MM)
    ap.add_argument("--short", type=float, default=calibration.A4_SHORT_MM)
    ap.add_argument("--out", default="", help="标注图输出路径，默认同目录 diag_*.jpg")
    args = ap.parse_args()

    img = read_image(args.image)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        print("一个轮廓都没有：背景不比纸暗")
        return

    paper = max(contours, key=cv2.contourArea)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)[:4]
    rect = cv2.minAreaRect(paper)
    (_cx, _cy), (rw, rh), ang = rect
    long_px, short_px = max(rw, rh), min(rw, rh)
    box = cv2.boxPoints(rect)
    fill = cv2.contourArea(paper) / (rw * rh) if rw * rh > 0 else 0.0

    print(f"画面 {w}x{h} | Otsu 阈值 {thr:.0f} | 亮区占画面 "
          f"{cv2.countNonZero(binary) / (w * h) * 100:.1f}%")
    print(f"轮廓数 {len(contours)} | 前几大轮廓占画面% "
          f"{[round(a / (w * h) * 100, 1) for a in areas]}")
    print(f"外接矩形 {long_px:.0f}x{short_px:.0f}px 角度 {ang:.1f}° | "
          f"轮廓/矩形填充率 {fill * 100:.1f}%（矩形纸应 >95%）")
    print(f"ppm_long {long_px / args.long:.4f} | ppm_short {short_px / args.short:.4f} | "
          f"互差 {abs(long_px / args.long - short_px / args.short) / (long_px / args.long) * 100:.2f}%")
    print(f"长宽比 {long_px / short_px:.4f}（A4 应为 {calibration.A4_ASPECT:.4f}）")
    print(f"矩形角点 x {box[:, 0].min():.0f}~{box[:, 0].max():.0f} (画面 0~{w}) | "
          f"y {box[:, 1].min():.0f}~{box[:, 1].max():.0f} (画面 0~{h})")
    print(f"纸的像素边长 {long_px:.0f}px = {args.long}mm → 1px = "
          f"{1000 * args.long / long_px:.1f} µm")

    ratio = long_px / short_px
    diff = abs(long_px / args.long - short_px / args.short) / (long_px / args.long)
    bright = cv2.countNonZero(binary) / (w * h)
    stray = bright - areas[0] / (w * h)
    margin = (box[:, 0].min() > 20 and box[:, 1].min() > 20
              and box[:, 0].max() < w - 20 and box[:, 1].max() < h - 20)
    checks = [
        ('轮廓/矩形填充率 > 97%', fill > 0.97, f'{fill * 100:.1f}%'),
        (f'长宽比 {calibration.A4_ASPECT:.4f} ±0.5%',
         abs(ratio - calibration.A4_ASPECT) / calibration.A4_ASPECT <= 0.005, f'{ratio:.4f}'),
        ('长短边 ppm 互差 < 1%', diff < 0.01, f'{diff * 100:.2f}%'),
        ('外接矩形四角都在画面内', margin,
         f"x {box[:, 0].min():.0f}~{box[:, 0].max():.0f} y {box[:, 1].min():.0f}~{box[:, 1].max():.0f}"),
        ('亮区几乎全是纸（杂亮 < 2%）', stray < 0.02, f'{stray * 100:.1f}%'),
    ]
    print('判卷：')
    for name, ok, value in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} 实测 {value}")
    print('结论：' + ('合格，可以拿去标定' if all(c[1] for c in checks)
                     else '不合格，按上面 FAIL 项重拍'))

    vis = cv2.resize(img, (1000, int(h * 1000 / w)))
    scale = 1000 / w
    cv2.drawContours(vis, [np.round(paper * scale).astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.drawContours(vis, [np.round(box * scale).astype(np.int32)], 0, (0, 0, 255), 2)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    ascii_stem = re.sub(r'[^0-9A-Za-z_.-]', '', stem)[-24:] or 'a4'   # 中文名 cv2.imwrite 不保证能写
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                   f"diag_{ascii_stem}.jpg")
    if not cv2.imwrite(out, vis):
        raise RuntimeError(f"标注图写盘失败: {out}")
    print(f"标注图（绿=选中轮廓，红=外接矩形）: {out}")


if __name__ == "__main__":
    main()
