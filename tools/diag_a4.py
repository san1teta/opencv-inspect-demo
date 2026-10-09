"""把程序"看到的"东西画出来：Otsu 二值化、选中的轮廓、外接矩形，缩图存盘。

标定不合格时先看这张图，别猜。
"""
import argparse
import io
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import calibration  # noqa: E402
import image_io  # noqa: E402

# 中文 Windows 控制台默认 GBK，指标里混进一个编不出的字符（µ、↔ 这类）就会抛 UnicodeEncodeError
# 把整份诊断带走。宁可打成 ? 也要把结论吐完。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(errors='replace')


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 标定诊断")
    ap.add_argument("image")
    ap.add_argument("--long", type=float, default=calibration.A4_LONG_MM)
    ap.add_argument("--short", type=float, default=calibration.A4_SHORT_MM)
    ap.add_argument("--out", default="", help="标注图输出路径，默认同目录 diag_*.jpg")
    args = ap.parse_args()

    img = image_io.read_image(args.image)
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
    expect_aspect = args.long / args.short   # 用实测边长算，别用名义 297/210

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [paper], -1, 255, -1)
    inside = gray[mask > 0]
    mean_in = float(inside.mean()) if inside.size else 0.0
    clipped = float((inside >= 255).mean() * 100) if inside.size else 0.0

    print('—— 以下是 Otsu 备用算子的内部指标，最终判卷以下方「标定算子」那行为准 ——')
    print(f"画面 {w}x{h} | Otsu 阈值 {thr:.0f} | 亮区占画面 "
          f"{cv2.countNonZero(binary) / (w * h) * 100:.1f}%")
    print(f"轮廓数 {len(contours)} | 前几大轮廓占画面% "
          f"{[round(a / (w * h) * 100, 1) for a in areas]}")
    print(f"外接矩形 {long_px:.0f}x{short_px:.0f}px 角度 {ang:.1f}° | "
          f"轮廓/矩形填充率 {fill * 100:.1f}%（矩形纸应 >95%）")
    print(f"ppm_long {long_px / args.long:.4f} | ppm_short {short_px / args.short:.4f} | "
          f"互差 {abs(long_px / args.long - short_px / args.short) / (long_px / args.long) * 100:.2f}%")
    print(f"长宽比 {long_px / short_px:.4f}（本纸应为 {expect_aspect:.4f}）")
    print(f"纸面平均灰度 {mean_in:.0f}（宜 150~240）| 饱和=255 占比 {clipped:.1f}%"
          f"（宜 <5%，合成纯色图必然 100%，只看实拍）")
    print(f"矩形角点 x {box[:, 0].min():.0f}~{box[:, 0].max():.0f} (画面 0~{w}) | "
          f"y {box[:, 1].min():.0f}~{box[:, 1].max():.0f} (画面 0~{h})")
    print(f"纸的像素边长 {long_px:.0f}px = {args.long}mm → 1px = "
          f"{1000 * args.long / long_px:.1f} 微米")   # 别用 µ，GBK 控制台一接管道就崩

    bright = cv2.countNonZero(binary) / (w * h)
    stray = bright - areas[0] / (w * h)
    m = calibration.measure_a4(img, args.long, args.short, 99.0)   # 放宽阈值拿全部数字，判卷在下面自己下
    m_diff = abs(m['ppm_long'] - m['ppm_short']) / m['ppm_long']
    # 裁角风险要按「纸在哪个轴上就除以那个轴」算。原先写 long_px / max(w, h)：
    # 竖构图里纸的长边是横着摊开的，它却拿画面最长边（竖向 4096）去除，
    # 于是报 72.4%「留白充足」，而真实横向占比是 96.4%、每侧只剩 5.5mm ——
    # 差 24 个百分点，而且方向是「把危险报成安全」，正是本项目最忌的静默错值。
    ext_x = float(box[:, 0].max() - box[:, 0].min())
    ext_y = float(box[:, 1].max() - box[:, 1].min())
    span = max(ext_x / w, ext_y / h)
    margin_mm = min((w - ext_x) / 2, (h - ext_y) / 2) / m['ppm']
    print(f"纸占画面 横向 {ext_x / w * 100:.1f}% / 纵向 {ext_y / h * 100:.1f}%（取较大者判裁角风险）"
          f" | 最紧一侧每侧留白 {margin_mm:.1f}mm —— 挪机时不许超过这个数")
    grad = 0.0
    if len(m['sides']) == 4:   # 只有四边形算子给得出逐边长度，Otsu 兜底时退化为外接矩形（梯度恒 0）
        s = m['sides']
        grad = max(abs(s[0] - s[2]) / ((s[0] + s[2]) / 2), abs(s[1] - s[3]) / ((s[1] + s[3]) / 2))
    checks = [
        ('FAIL', f"长宽比 {expect_aspect:.4f} ±0.5%",
         abs(m['aspect'] - expect_aspect) / expect_aspect <= 0.005, f"{m['aspect']:.4f}"),
        ('FAIL', '长短边 ppm 互差 < 1%', m_diff < 0.01, f'{m_diff * 100:.2f}%'),
        ('WARN', '轮廓/矩形填充率 > 97%', fill > 0.97, f'{fill * 100:.1f}%'),
        ('WARN', '亮区几乎全是纸（杂亮 < 2%）', stray < 0.02, f'{stray * 100:.1f}%'),
        ('WARN', '纸占画面（两轴取大）< 95%（太满易裁角）', span < 0.95, f'{span * 100:.1f}%'),
        ('WARN', '纸面平均灰度 150~240', 150 <= mean_in <= 240, f'{mean_in:.0f}'),
        ('WARN', '纸面饱和=255 占比 < 5%', clipped < 5.0, f'{clipped:.1f}%'),
        ('WARN', '比例尺梯度（对边差）< 0.5%', grad < 0.005, f'{grad * 100:.2f}%'),
    ]
    cx, cy = m['center']
    print(f"标定算子：{m['method']} | ppm_long {m['ppm_long']:.4f} | ppm_short {m['ppm_short']:.4f}"
          f" | 采用 ppm {m['ppm']:.4f}")
    print(f"纸心 ({cx:.0f}, {cy:.0f}) | 画面中心 ({w // 2}, {h // 2}) —— 纸心在画面中心"
          f"{'上' if cy < h / 2 else '下'} {abs(cy - h / 2) / m['ppm']:.0f}mm，"
          f"待测物要对准纸心而不是画面中心（tools/check_placement.py 会量给你看）")
    if grad:
        # grad 是对边峰峰值，纸心↔纸边只吃一半；别把峰峰值当成中心偏差报，会虚高一倍
        half = grad * 25 / 2
        print(f"比例尺梯度 {grad * 100:.2f}%（对边峰峰值）：ppm 取四边均值，只在纸心严格成立。"
              f"按 25mm 直径：纸心到纸边 ±{half:.3f}mm，纸两端之间 {grad * 25:.3f}mm")
        print(f"  → 待测物放纸心 ±{0.05 * args.short / (grad * 25):.0f}mm 内，该项即可压到 0.05mm 以下")
    print('判卷（WARN 只是提醒，不影响结论；杂亮高但长宽比/互差都过 = 杂物没粘连，可用）：')
    for level, name, ok, value in checks:
        print(f"  {'PASS' if ok else level:<4} {name:<34} 实测 {value}")
    hard_fail = any(not c[2] for c in checks if c[0] == 'FAIL')
    print('结论：' + ('不合格，按上面 FAIL 项重拍' if hard_fail else '合格，可以拿去标定'))

    vis = cv2.resize(img, (1000, int(h * 1000 / w)))
    scale = 1000 / w
    cv2.drawContours(vis, [np.round(paper * scale).astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.drawContours(vis, [np.round(box * scale).astype(np.int32)], 0, (0, 0, 255), 2)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    ascii_stem = re.sub(r'[^0-9A-Za-z_.-]', '', stem)[-24:] or 'a4'   # 中文名 cv2.imwrite 不保证能写
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                   f"diag_{ascii_stem}.jpg")
    if not image_io.write_image(out, vis):
        raise RuntimeError(f"标注图写盘失败: {out}")
    print(f"标注图（绿=选中轮廓，红=外接矩形）: {out}")


if __name__ == "__main__":
    main()
