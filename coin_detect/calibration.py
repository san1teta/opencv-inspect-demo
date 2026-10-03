"""像素 ↔ 毫米换算 + 用 A4 白纸做的现场标定（标定物 = 一张 ISO 216 A4 纸）。"""
import cv2
import numpy as np

A4_LONG_MM = 297.0
A4_SHORT_MM = 210.0
A4_ASPECT = A4_LONG_MM / A4_SHORT_MM   # √2 ≈ 1.41421


def pixels_to_mm(pixels: float, ppm: float) -> float:
    return pixels / ppm


def measure_a4(img: np.ndarray, long_mm: float = A4_LONG_MM, short_mm: float = A4_SHORT_MM,
               tol: float = 0.01) -> dict[str, float]:
    """量出画面里那张 A4 纸，返回两套 ppm 和长宽比；长短边互差超 tol 直接 raise。

    硬币压在纸上不影响结果：RETR_EXTERNAL 只取最外层轮廓，纸把硬币整个包在里面。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        raise ValueError("画面里没有亮区，找不到 A4 纸（桌面必须比纸暗，否则 Otsu 分不开）")

    paper = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _angle = cv2.minAreaRect(paper)
    long_px, short_px = max(w, h), min(w, h)
    if short_px < 20:
        raise ValueError(f"纸的短边只有 {short_px:.0f}px，太小不可信")

    ppm_long = float(long_px / long_mm)
    ppm_short = float(short_px / short_mm)
    if abs(ppm_long - ppm_short) / ppm_long > tol:
        raise ValueError(f"长短边 ppm 不一致：{ppm_long:.3f} vs {ppm_short:.3f}"
                         f"（透视没摆正 / 纸角出画面 / 桌面比纸亮）")

    return {
        'ppm': float((ppm_long + ppm_short) / 2),
        'ppm_long': ppm_long,
        'ppm_short': ppm_short,
        'aspect': float(long_px / short_px),
        'long_px': float(long_px),
        'short_px': float(short_px),
    }


def measure_ppm_from_a4(img: np.ndarray, long_mm: float = A4_LONG_MM,
                        short_mm: float = A4_SHORT_MM, tol: float = 0.01) -> float:
    """只要一个数：这张照片对应的 ppm（像素/毫米）。"""
    return measure_a4(img, long_mm, short_mm, tol)['ppm']

