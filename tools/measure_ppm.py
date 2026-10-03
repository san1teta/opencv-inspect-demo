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

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import calibration  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'coin_detect', 'config_setting.py'))


def write_ppm(config_path: str, ppm: float) -> str:
    """把 ppm 写进 config_setting.py 的那一行，返回替换后的整行文本。"""
    with open(config_path, encoding='utf-8') as f:
        text = f.read()
    line = f"ppm = {ppm:.4f}  # A4 实测于 {datetime.date.today().isoformat()}，旧值 10.0 是演示假定值"
    new_text, n = re.subn(r'(?m)^ppm\s*=.*$', line, text, count=1)
    if n == 0:
        raise ValueError(f"{config_path} 里找不到 ppm 那一行，没敢改")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    return line


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

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"读不到图片: {args.image}")

    m = calibration.measure_a4(img, args.long, args.short, args.tol)
    logger.info(f"长边 {m['long_px']:.1f}px → {m['ppm_long']:.4f} ppm | "
                f"短边 {m['short_px']:.1f}px → {m['ppm_short']:.4f} ppm")
    logger.info(f"长宽比 {m['aspect']:.4f}（A4 应为 {calibration.A4_ASPECT:.4f}，"
                f"差 {abs(m['aspect'] - calibration.A4_ASPECT) / calibration.A4_ASPECT * 100:.2f}%）")
    logger.info(f"ppm = {m['ppm']:.4f}  （即 1 px = {1000 / m['ppm']:.1f} µm）")

    if args.write:
        logger.info("已写入: " + write_ppm(args.config, m['ppm']))


if __name__ == "__main__":
    main()
