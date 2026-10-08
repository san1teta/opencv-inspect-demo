"""用一张 A4 纸的照片标定 ppm，可一条命令写回 config_setting.py。

用法：
    python tools/measure_ppm.py <照片路径>                     # 只报数
    python tools/measure_ppm.py <照片路径> --long 296.5 --short 209.8   # 用钢尺实测值
    python tools/measure_ppm.py <照片路径> --write              # 写进 config_setting.py
"""
import argparse
import datetime
import logging
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import calibration  # noqa: E402
import image_io  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'coin_detect', 'config_setting.py'))


def write_ppm(config_path: str, ppm: float, img_size: tuple[int, int],
              center: tuple[float, float]) -> list[str]:
    """把 ppm、标定图尺寸和纸心写进 config_setting.py，返回改动后的行文本。"""
    with open(config_path, encoding='utf-8') as f:
        text = f.read()
    today = datetime.date.today().isoformat()
    targets = [
        (r'(?m)^ppm\s*=',
         f"ppm = {ppm:.4f}  # A4 实测于 {today}，旧值 10.0 是演示假定值"),
        (r'(?m)^calib_image_size\s*=',
         f"calib_image_size = ({img_size[0]}, {img_size[1]})"
         f"  # 标定图宽高，测量图必须与此一致"),
        (r'(?m)^calib_center\s*=',
         f"calib_center = ({center[0]:.1f}, {center[1]:.1f})"
         f"  # 标定时的纸心 xy —— ppm 是四边均值，只在这个位置严格成立；"
         f"注意它不等于画面中心"),
    ]
    changed: list[str] = []
    for pattern, line in targets:
        text, n = re.subn(pattern + r'.*$', line, text, count=1)
        if n == 0:
            raise ValueError(f"{config_path} 里找不到 {pattern[4:-2]} 那一行，没敢改")
        changed.append(line)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return changed


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description="用 A4 纸标定 ppm")
    parser.add_argument("image", help="含一张平整 A4 纸的照片（桌面要比纸暗）")
    parser.add_argument("--long", type=float, default=calibration.A4_LONG_MM, help="钢尺实测长边 mm")
    parser.add_argument("--short", type=float, default=calibration.A4_SHORT_MM, help="钢尺实测短边 mm")
    parser.add_argument("--tol", type=float, default=0.01, help="长短边 ppm 允许的相对偏差")
    parser.add_argument("--write", action="store_true", help="把结果写进 config_setting.py")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    args = parser.parse_args()

    try:
        img = image_io.read_image(args.image)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    try:
        m = calibration.measure_a4(img, args.long, args.short, args.tol)
    except ValueError as exc:
        sys.exit(f"标定不合格：{exc}")

    h, w = img.shape[:2]
    logger.info(f"标定图尺寸 {w}x{h}px —— 之后所有待测图必须是同一尺寸，否则 ppm 不适用")
    logger.info(f"长边 {m['long_px']:.1f}px → {m['ppm_long']:.4f} ppm | "
                f"短边 {m['short_px']:.1f}px → {m['ppm_short']:.4f} ppm")
    expect_aspect = args.long / args.short
    logger.info(f"长宽比 {m['aspect']:.4f}（本纸应为 {expect_aspect:.4f}，"
                f"差 {abs(m['aspect'] - expect_aspect) / expect_aspect * 100:.2f}%）")
    logger.info(f"ppm = {m['ppm']:.4f}  （即 1 px = {1000 / m['ppm']:.1f} 微米）")

    cx, cy = m['center']
    logger.info(f"纸心 ({cx:.0f}, {cy:.0f}) | 画面中心 ({w // 2}, {h // 2}) —— "
                f"两者相差 {abs(cx - w / 2) / m['ppm']:.0f}x{abs(cy - h / 2) / m['ppm']:.0f}mm，"
                f"待测物要对准的是纸心，不是画面中心")

    if args.write:
        for line in write_ppm(args.config, m['ppm'], (w, h), (cx, cy)):
            logger.info("已写入: " + line)


if __name__ == "__main__":
    main()
