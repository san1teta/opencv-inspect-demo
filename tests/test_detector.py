"""detector 的估计量必须对着「已知真值的圆」来量，光看跑不跑得通没用。

一律走真实预处理链（高斯 + Canny），因为偏差恰恰出在 Canny 环的外多边形上，
直接喂一个理想轮廓是测不出来的。
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import cv2
import numpy as np

import config_setting
import detector
import preprocess

PPM = 9.9749   # 2026-10-08 A4 实测标定值，1px ≈ 0.1mm


def _measure(diameter_mm: float, nick_mm: float = 0.0) -> tuple[dict, float]:
    """画一枚直径已知的硬币，返回（最大轮廓特征, 实际画出来的直径 mm）。

    cv2.circle 的半径只能取整，所以真值是「画出来的」那个直径，不是传进去的。
    nick_mm 是从边缘啃掉的缺口深度（咬痕圆的半径 = 缺口深度）。
    """
    img = np.full((700, 700), 40, np.uint8)
    r = int(round(diameter_mm * PPM / 2))
    cv2.circle(img, (350, 350), r, 200, -1)
    if nick_mm > 0:
        # 圆心压在轮廓上、半径 = 深度，所以啃进去正好 nick_mm（与 selftest_measure 同一约定）
        cv2.circle(img, (350 + r, 350), int(round(nick_mm * PPM)), 40, -1)
    edges = preprocess.canny(
        preprocess.gaussian_blur(img, config_setting.gaussian_kernel),
        config_setting.canny_low, config_setting.canny_high)
    cnt = max(detector.find_contours(edges, config_setting.min_area), key=cv2.contourArea)
    return detector.calculate_features(cnt), 2 * r / PPM


class TestCalculateFeatures(unittest.TestCase):
    def test_diameter_has_no_systematic_bias(self):
        """曾经用外接圆直径，19/20.5/25mm 一律 +1.21px（+0.12mm），吃光整个精度预算。"""
        for d in (19.0, 20.5, 25.0):
            with self.subTest(diameter=d):
                f, drawn = _measure(d)
                self.assertLess(abs(f['diameter_pixels'] / PPM - drawn), 0.05)

    def test_perfect_circle_clears_the_gate(self):
        """完美圆的 4πA/P² 只有 0.899，低于门槛 0.9 → 任何硬币都会被判不合格。"""
        f, _ = _measure(25.0)
        self.assertGreater(f['circularity'], config_setting.min_circularity)

    def test_intact_coin_sits_on_the_deficit_noise_floor(self):
        """完整圆的径向凹陷必须贴着噪声底，否则刀口判据会误报成真币缺陷。"""
        for d in (19.0, 25.0):
            with self.subTest(diameter=d):
                f, _ = _measure(d)
                self.assertLess(f['deficit_pixels'] / PPM, 0.15)

    def test_deficit_measures_notch_depth(self):
        """径向凹陷要定量：报出来的数就该等于实际啃掉的深度。"""
        for depth in (0.5, 1.0, 2.0):
            with self.subTest(depth=depth):
                f, _ = _measure(25.0, nick_mm=depth)
                self.assertAlmostEqual(f['deficit_pixels'] / PPM, depth, delta=0.15)

    def test_notch_is_invisible_to_size_but_not_to_edge(self):
        """换判据的全部理由：2mm 深的缺口只让等效直径小 0.15mm —— 比允差 0.25mm 还小，
        尺寸路必然判合格；而缺损路直接报出 2mm 并给档。阈值挂在允差上，不挂在我随手写的数上。
        """
        intact, _ = _measure(25.0)
        nicked, _ = _measure(25.0, nick_mm=2.0)
        tol = config_setting.coin_specs['1元']['tol']
        self.assertLess(abs(intact['diameter_pixels'] - nicked['diameter_pixels']) / PPM, tol)
        self.assertGreater(nicked['deficit_pixels'] / PPM, config_setting.defect_mid)

    def test_overall_wear_is_not_reported_as_a_notch(self):
        """整体磨小是中位半径一起缩，凹陷仍是噪声底 —— 尺寸路和缺损路不能互相污染。"""
        big, _ = _measure(25.0)
        worn, _ = _measure(23.5)
        self.assertLess(worn['diameter_pixels'], big['diameter_pixels'])
        self.assertLess(worn['deficit_pixels'], 0.15 * PPM)


if __name__ == '__main__':
    unittest.main()
