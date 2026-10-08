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


def _measure(diameter_mm: float, nick: bool = False) -> tuple[dict, float]:
    """画一枚直径已知的硬币，返回（最大轮廓特征, 实际画出来的直径 mm）。

    cv2.circle 的半径只能取整，所以真值是「画出来的」那个直径，不是传进去的。
    """
    img = np.full((700, 700), 40, np.uint8)
    r = int(round(diameter_mm * PPM / 2))
    cv2.circle(img, (350, 350), r, 200, -1)
    if nick:                       # 刀口：从边缘啃掉一块
        cv2.circle(img, (350 + r, 350), max(2, r // 8), 40, -1)
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

    def test_edge_loss_shows_up_in_diameter(self):
        """检的就是边缘缺损，所以啃掉一块之后直径必须变小；外接圆做不到这一点。"""
        intact, _ = _measure(25.0)
        nicked, _ = _measure(25.0, nick=True)
        self.assertLess(nicked['diameter_pixels'], intact['diameter_pixels'])
        self.assertLess(nicked['circularity'], intact['circularity'])


if __name__ == '__main__':
    unittest.main()
