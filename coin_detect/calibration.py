"""像素 ↔ 毫米换算 + 用 A4 白纸做的现场标定（标定物 = 一张 ISO 216 A4 纸）。"""
from typing import TypedDict

import cv2
import numpy as np

A4_LONG_MM = 297.0
A4_SHORT_MM = 210.0
A4_ASPECT = A4_LONG_MM / A4_SHORT_MM   # √2 ≈ 1.41421


def pixels_to_mm(pixels: float, ppm: float) -> float:
    return pixels / ppm


def quad_center(p: np.ndarray) -> tuple[float, float]:
    """四边形两条对角线的交点 —— 它就是原矩形物理中心的像。

    为什么不能用四个顶点的平均值：射影变换保持共线与共点。矩形的两条对角线交于物理中心，
    变换后仍是这个四边形的两条对角线，所以**交点 = 物理中心的像**，这是唯一与几何定义
    一致的估计量。顶点平均落在「像素中点」上，而纸的近端占了更多像素，于是顶点平均被
    系统性拉向近边。

    本装置 2026-10-10 实测：顶点平均 (1529.5, 1424.2) vs 交点 (1536.9, 1421.1)，
    差 **0.93mm**（左右 0.86 / 上下 0.36，与长边对 0.70%、短边对 1.16% 的梯度方向一致 ——
    左边离镜头更近，所以真纸心偏右）。对直径测量本身影响可忽略（那点距离上 ppm 只变
    约 0.0008mm），但 `calib_center` 这个名字承诺的是「纸心」，就该给出纸心：币压在
    直尺找出的纸心上时，`check_placement` 报出的「偏离纸心」不该含这 0.93mm 系统量。
    """
    a1, a2 = p[0], p[2]
    b1, b2 = p[1], p[3]
    d1, d2 = a2 - a1, b2 - b1
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) < 1e-9:                 # 对角线平行 = 四边形退化，退回顶点平均而不是崩掉
        c = np.asarray(p, dtype=np.float64).mean(axis=0)
        return float(c[0]), float(c[1])
    t = float(((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den)
    return float(a1[0] + t * d1[0]), float(a1[1] + t * d1[1])


def quad_sides(gray: np.ndarray) -> tuple[list[float], tuple[float, float]] | None:
    """边缘 + 四边形拟合找纸：主动忽略杂物（凳腿/木纹/锁边线都不是大四边形）。

    返回 (按顶点顺序的 4 条边长, 四边形中心 xy)，找不到可信四边形时返回 None。
    边长原样吐出来而不是只吐长短边，是因为「对边是否相等」本身就是透视指标
    （见 quad_pair_means / diag_a4 的比例尺梯度）。中心点则是「比例尺在哪儿成立」的
    唯一参考位置 —— ppm 是四边均值，只在这个点上严格适用。
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
        if min(quad_pair_means(sides)) < 20:
            continue
        c = quad_center(p)
        return sides, c
    return None


def quad_pair_means(sides: list[float]) -> tuple[float, float]:
    """对边取平均折成长短边，以削弱透视影响。返回 (长边px, 短边px)。"""
    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    return max(a, b), min(a, b)


def _bright_side_lengths(binary: np.ndarray) -> tuple[float, float, float, tuple[float, float]]:
    """备用算子：Otsu 二值化后取最大亮区的外接矩形。返回 (长边px, 短边px, 填充率, 中心xy)。"""
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        raise ValueError("画面里没有亮区，找不到 A4 纸（背景必须比纸暗）")
    paper = max(contours, key=cv2.contourArea)
    (cx, cy), (rw, rh), _angle = cv2.minAreaRect(paper)
    fill = cv2.contourArea(paper) / (rw * rh) if rw * rh > 0 else 0.0
    return max(rw, rh), min(rw, rh), fill, (float(cx), float(cy))


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
    sides: list[float]   # 四边形算子的 4 条边长（px），Otsu 兜底时为空
    center: list[float]  # 纸心 xy —— ppm 只在这个位置严格成立


def measure_a4(img: np.ndarray, long_mm: float = A4_LONG_MM, short_mm: float = A4_SHORT_MM,
               tol: float = 0.01) -> A4Measurement:
    """量出画面里那张 A4 纸，返回两套 ppm、长宽比和所用算子；长短边互差超 tol 直接 raise。

    先走四边形拟合（能主动忽略杂物），失败再退回 Otsu 最大亮区。
    img 允许彩色或单通道灰度，两者结果必须一致。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    quad = quad_sides(gray)
    if quad is not None:
        sides, center = quad
        long_px, short_px = quad_pair_means(sides)
        fill, method = 1.0, 'quad'
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        long_px, short_px, fill, center = _bright_side_lengths(binary)
        sides, method = [], 'otsu'

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
        'sides': sides,
        'center': [center[0], center[1]],
    }


def measure_ppm_from_a4(img: np.ndarray, long_mm: float = A4_LONG_MM,
                        short_mm: float = A4_SHORT_MM, tol: float = 0.01) -> float:
    """只要一个数：这张照片对应的 ppm（像素/毫米）。"""
    return measure_a4(img, long_mm, short_mm, tol)['ppm']

