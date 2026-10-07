"""像素 ↔ 毫米换算 + 用 A4 白纸做的现场标定（标定物 = 一张 ISO 216 A4 纸）。"""
from typing import TypedDict

import cv2
import numpy as np

A4_LONG_MM = 297.0
A4_SHORT_MM = 210.0
A4_ASPECT = A4_LONG_MM / A4_SHORT_MM   # √2 ≈ 1.41421


def pixels_to_mm(pixels: float, ppm: float) -> float:
    return pixels / ppm


def _quad_side_lengths(gray: np.ndarray) -> tuple[float, float] | None:
    """边缘 + 四边形拟合找纸：主动忽略杂物（凳腿/木纹/锁边线都不是大四边形）。

    返回 (长边px, 短边px)，找不到可信四边形时返回 None。对边取平均以削弱透视影响。
    """
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))   # 把断掉的纸边接上
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or cv2.contourArea(approx) < 0.1 * w * h:
            continue
        p = approx.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(p[i] - p[(i + 1) % 4])) for i in range(4)]
        a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
        if min(a, b) < 20:
            continue
        return max(a, b), min(a, b)
    return None


def _bright_side_lengths(binary: np.ndarray) -> tuple[float, float, float]:
    """备用算子：Otsu 二值化后取最大亮区的外接矩形。返回 (长边px, 短边px, 填充率)。"""
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        raise ValueError("画面里没有亮区，找不到 A4 纸（背景必须比纸暗）")
    paper = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _angle = cv2.minAreaRect(paper)
    fill = cv2.contourArea(paper) / (rw * rh) if rw * rh > 0 else 0.0
    return max(rw, rh), min(rw, rh), fill


class A4Measurement(TypedDict):
    """measure_a4 的返回值：ppm 用 float 存，算子名用 str 存，不让 Any 传染下游。"""
    ppm: float
    ppm_long: float
    ppm_short: float
    aspect: float
    long_px: float
    short_px: float
    fill: float
    method: str


def measure_a4(img: np.ndarray, long_mm: float = A4_LONG_MM, short_mm: float = A4_SHORT_MM,
               tol: float = 0.01) -> A4Measurement:
    """量出画面里那张 A4 纸，返回两套 ppm、长宽比和所用算子；长短边互差超 tol 直接 raise。

    先走四边形拟合（能主动忽略杂物），失败再退回 Otsu 最大亮区。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = _quad_side_lengths(gray)
    if quad is not None:
        long_px, short_px, fill, method = quad[0], quad[1], 1.0, 'quad'
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        long_px, short_px, fill = _bright_side_lengths(binary)
        method = 'otsu'

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
        'fill': float(fill),
        'method': method,
    }


def measure_ppm_from_a4(img: np.ndarray, long_mm: float = A4_LONG_MM,
                        short_mm: float = A4_SHORT_MM, tol: float = 0.01) -> float:
    """只要一个数：这张照片对应的 ppm（像素/毫米）。"""
    return measure_a4(img, long_mm, short_mm, tol)['ppm']

