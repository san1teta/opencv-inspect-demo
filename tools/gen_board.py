"""生成棋盘格标定板图（一次性工具，不参与检测链路）。

两版共用 10×7 方格 = 9×6 内角点，换板子不用改 calibrateCamera 的 pattern。
方格物理边长一律以钢尺实测为准，名义值只用来选纸张。
"""
import argparse
import ctypes
import logging
import os
import sys
from ctypes import wintypes
from typing import TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COLS = 10                      # 方格列数
ROWS = 7                       # 方格行数
INNER = (COLS - 1, ROWS - 1)   # 内角点数，最外圈方格没有内侧交点

TITLEBAR_PX = 32  # OpenCV HighGUI 窗口标题栏占高，Windows 10 约 31px
SPI_GETWORKAREA = 0x30  # SystemParametersInfoW：取已扣掉任务栏的工作区


class Preset(TypedDict):
    square_px: int             # 每个方格的图像像素边长
    margin_px: int             # 四周白边宽度（SB 检测器要求有包边）
    dpi: int | None            # 写入 PNG 的 pHYs，None 表示不写
    square_mm: float | None    # 名义物理边长，实测值才作数
    hint: str


# square_mm 是名义值，打印/显示后必须用钢尺复核
PRESETS: dict[str, Preset] = {
    'screen': {'square_px': 100, 'margin_px': 100, 'dpi': None, 'square_mm': None,
               'hint': '笔记本屏幕 1:1 显示，配 --show 用；量到的毫米数 ÷ 10 才是真 square_mm'},
    'print': {'square_px': 295, 'margin_px': 118, 'dpi': 300, 'square_mm': 25.0,
              'hint': 'A4 横向、25mm/格、缩放选 100%/实际大小；打印店取件先用钢尺复核'},
}


def make_checkerboard(square_px: int, cols: int, rows: int, margin_px: int) -> np.ndarray:
    """画 cols×rows 个方格的黑白棋盘格，四周留 margin_px 白边，返回 BGR 图。"""
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    cells = ((r + c) % 2 * 255).astype(np.uint8)
    board = cv2.resize(cells, (cols * square_px, rows * square_px),
                       interpolation=cv2.INTER_NEAREST)
    board = cv2.copyMakeBorder(board, margin_px, margin_px, margin_px, margin_px,
                               cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def count_inner_corners(img: np.ndarray) -> int:
    """自己生成的图先自检：SB 检测器能找到几个内角点。"""
    ok, corners = cv2.findChessboardCornersSB(img, INNER)
    return len(corners) if ok else 0


def save_png(img: np.ndarray, path: str, dpi: int | None) -> bool:
    """存 PNG；带 dpi 时补 pHYs 元数据，否则打印软件会按 72dpi 猜尺寸。"""
    if dpi is None:
        return cv2.imwrite(path, img)
    try:
        from PIL import Image
    except ImportError:
        logger.warning("本机没有 Pillow，改存无 DPI 元数据的版本（打印时须手动选 100% 缩放）")
        return cv2.imwrite(path, img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Image.fromarray(gray).save(path, dpi=(dpi, dpi))
    logger.info(f"已写入 {dpi}dpi 元数据，图片尺寸 {w}x{h}px")
    return True


def make_process_dpi_aware() -> None:
    """让 1 个图像像素 = 1 个物理像素，否则 Windows 缩放会把板子拉伸。"""
    if not hasattr(ctypes.windll, 'user32'):
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()


def _work_area() -> tuple[int, int]:
    """主显示器工作区像素尺寸（已扣任务栏）。

    GetSystemMetrics(78/79) 本机实测返回全屏 1920x1080，不可信 → 必须走 SPI_GETWORKAREA。
    """
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        sys.exit("读不到工作区尺寸，无法校验板子会不会被裁切")
    return int(rect.right), int(rect.bottom)


def show_1to1(img: np.ndarray, title: str) -> None:
    """按原始像素尺寸显示（WINDOW_AUTOSIZE 不缩放），q 退出。"""
    h, w = img.shape[:2]
    aw, ah = _work_area()
    if w > aw or h + TITLEBAR_PX > ah:
        sys.exit(f"板子 {w}x{h}px 放不下工作区 {aw}x{ah}px（标题栏另占 {TITLEBAR_PX}px），"
                 f"底部方格会被切掉 → 把该 preset 的 square_px 改小再生成")
    make_process_dpi_aware()
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    logger.info("屏幕显示中：请确认窗口未被缩放、未贴边裁切，按 q 关闭")
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description="生成棋盘格标定板")
    parser.add_argument("--preset", choices=list(PRESETS), default="screen")
    parser.add_argument("--out", type=str, default=os.path.join("sample_test", "calibration"))
    parser.add_argument("--show", action="store_true", help="1:1 显示用于屏幕标定")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    img = make_checkerboard(cfg['square_px'], COLS, ROWS, cfg['margin_px'])

    found = count_inner_corners(img)
    expect = INNER[0] * INNER[1]
    logger.info(f"自检：内角点 {found}/{expect}（{INNER[0]}x{INNER[1]}）")
    if found != expect:
        sys.exit(f"自检失败：白边或方格数有问题，这张板子不能用（{found} != {expect}）")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"board_{args.preset}_{COLS}x{ROWS}.png")
    if not save_png(img, path, cfg['dpi']):
        sys.exit(f"写盘失败：{path}")
    logger.info(f"标定板已生成：{path}  尺寸 {img.shape[1]}x{img.shape[0]}px")
    if cfg['square_mm'] is not None and cfg['dpi'] is not None:
        w_mm = img.shape[1] * 25.4 / cfg['dpi']
        h_mm = img.shape[0] * 25.4 / cfg['dpi']
        logger.info(f"名义方格 {cfg['square_mm']}mm → 整板约 {w_mm:.1f}x{h_mm:.1f}mm")
    logger.info(f"用法提示：{cfg['hint']}")
    logger.info("钢尺请量连续 10 个方格的总长再 ÷10，别量单格（读数误差会被放大 10 倍）")

    if args.show:
        show_1to1(img, f"checkerboard {COLS}x{ROWS}")


if __name__ == "__main__":
    main()
