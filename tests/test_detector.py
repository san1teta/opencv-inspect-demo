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


def _bucket(depth_mm: float) -> str:
    """缺损分级档位 —— 真正要守住的不是小数点，是落在哪一档。"""
    for name, lim in (('重度', config_setting.defect_heavy),
                      ('中度', config_setting.defect_mid),
                      ('轻度', config_setting.defect_light)):
        if depth_mm >= lim:
            return name
    return '合格'


def _spiked(diameter_mm: float, spike_px: int) -> tuple[dict, float]:
    """画一枚边缘带毛刺（向外凸起、不吞材料）的币，走同一条真实预处理链。"""
    img = np.full((700, 700), 40, np.uint8)
    r = int(round(diameter_mm * PPM / 2))
    cv2.circle(img, (350, 350), r, 200, -1)
    cv2.line(img, (350 + r - 4, 350), (350 + r + spike_px, 350), 200, 3)
    edges = preprocess.canny(preprocess.gaussian_blur(img, config_setting.gaussian_kernel),
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
        """径向凹陷要定量：报出来的数就该落在缺口真值附近，而且必须落进同一档。

        容差按相对误差 15% 给，不按绝对 0.15mm —— 质心会被啃掉的材料推向缺口反向，
        缺口越深低估越多（3mm 真值报 2.59mm）。分档才是这套判据真正要守住的东西。
        """
        for depth in (0.5, 1.0, 2.0, 3.0):
            with self.subTest(depth=depth):
                f, _ = _measure(25.0, nick_mm=depth)
                got = f['deficit_pixels'] / PPM
                self.assertLessEqual(abs(got - depth) / depth, 0.15)
                self.assertEqual(_bucket(got), _bucket(depth))

    def test_outward_burr_does_not_invent_a_notch(self):
        """回归锁：边缘一处向外毛刺不得凭空造出缺口。

        这是真币上踩到的坑 —— 完好 1元 量出 deficit 0.690mm，越过 defect_light=0.4，
        每一枚良品都会被误判成有缺口。根因是 deficit 曾绕着 minEnclosingCircle 的圆心算，
        而外接圆只由 2~3 个极端点决定，一处毛刺就把圆心拖走 5px。改用质心后同一枚币
        降到 0.148mm。合成完美圆测不出这条，因为完美圆的外接圆圆心恰好就是真圆心。
        """
        for spike in (0, 8, 16):
            with self.subTest(spike_px=spike):
                f, _ = _spiked(25.0, spike)
                self.assertLess(f['deficit_pixels'] / PPM, config_setting.defect_light)

    def test_center_is_the_centroid_not_the_envelope(self):
        """'center' 要落在币的几何中心：放位检查和缺陷径向都靠它，不能被极端点带偏。"""
        f, _ = _spiked(25.0, 16)
        self.assertLess(abs(f['center'][0] - 350), 3)
        self.assertLess(abs(f['center'][1] - 350), 3)


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


class TestDiameterWindow(unittest.TestCase):
    """闸门本身：像素窗口要等于「标称 ± gate」，两边都要算准。

    换算写在 detector 里，所以在这里锁死；写反一个符号（比如把 gate 加到直径上而不是
    标称两侧）在实拍上表现为「币全被毙掉」，而日志只说未检出，很难往回查。
    """

    def test_window_is_symmetric_around_nominal(self):
        lo, hi = detector.diameter_window(22.25, PPM, 2.0)
        self.assertAlmostEqual(lo / PPM, 20.25, places=6)
        self.assertAlmostEqual(hi / PPM, 24.25, places=6)

    def test_nominal_coin_falls_inside(self):
        f, _ = _measure(22.25)
        lo, hi = detector.diameter_window(22.25, PPM, config_setting.diameter_gate_mm)
        self.assertTrue(lo <= f['diameter_pixels'] <= hi,
                        f"标称尺寸的币被自己的闸门毙了：{f['diameter_pixels'] / PPM:.3f}mm")

    def test_reeding_shrinkage_still_clears_the_window(self):
        """实拍完好 1元 量出 22.07mm（比标称小 0.18mm，疑似齿边把等效直径拉到齿根侧）。
        闸门必须对这种系统性小偏完全无感，否则正常照片会被当成背景丢掉。"""
        f, _ = _measure(22.25 - 0.18)
        lo, hi = detector.diameter_window(22.25, PPM, config_setting.diameter_gate_mm)
        self.assertTrue(lo <= f['diameter_pixels'] <= hi)


if __name__ == '__main__':
    unittest.main()
