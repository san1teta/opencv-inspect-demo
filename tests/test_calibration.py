import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import cv2
import numpy as np

from calibration import pixels_to_mm, measure_a4, measure_ppm_from_a4


def synthetic_a4(ppm: float = 4.0, angle: float = 17.0, coin: bool = True) -> np.ndarray:
    """造一张「深色桌面 + 倾斜 A4 白纸 + 纸上压一枚硬币」的图，不依赖任何实拍素材。"""
    img = np.zeros((1400, 1000, 3), np.uint8)
    img[:, :] = (40, 40, 40)
    box = ((500, 700), (297 * ppm, 210 * ppm), angle)
    cv2.fillPoly(img, [cv2.boxPoints(box).astype(np.intp)], (255, 255, 255))
    if coin:
        cv2.circle(img, (500, 700), 60, (180, 180, 180), -1)
    return img


def _flat_paper(pts: list[list[int]]) -> np.ndarray:
    """深色桌面上画一块纸（顶点顺序任意，只要是个凸四边形）。"""
    img = np.full((1400, 1000, 3), 40, np.uint8)
    cv2.fillPoly(img, [np.array(pts, np.int32)], (255, 255, 255))
    return img


class TestPixelsToMm(unittest.TestCase):
    def test_pixels_to_mm(self):
        self.assertEqual(pixels_to_mm(100, 10.0), 10.0)

    def test_pixels_can_be_float(self):
        """detector 不再对直径像素取整，所以这里必须能吃小数。"""
        self.assertAlmostEqual(pixels_to_mm(100.5, 10.0), 10.05)


class TestMeasureA4(unittest.TestCase):
    def test_recovers_known_ppm(self):
        detail = measure_a4(synthetic_a4())
        self.assertAlmostEqual(detail['ppm'], 4.0, delta=0.02)
        self.assertAlmostEqual(measure_ppm_from_a4(synthetic_a4()), detail['ppm'])

    def test_coin_on_paper_does_not_change_result(self):
        """RETR_EXTERNAL 只取最外层轮廓，硬币在纸内部 → 不该影响长短边。"""
        with_coin = measure_a4(synthetic_a4(coin=True))
        no_coin = measure_a4(synthetic_a4(coin=False))
        self.assertAlmostEqual(with_coin['ppm'], no_coin['ppm'], places=6)

    def test_uses_measured_edge_lengths(self):
        """钢尺实测值必须真的参与运算：边长翻倍 → ppm 减半。"""
        detail = measure_a4(synthetic_a4(), long_mm=594.0, short_mm=420.0)
        self.assertAlmostEqual(detail['ppm'], 2.0, delta=0.01)

    def test_rejects_non_a4_rect(self):
        """白方块不是 A4，长短边会算出两个 ppm → 必须拦住而不是给个平均值。"""
        img = np.zeros((600, 600, 3), np.uint8)
        img[:, :] = (40, 40, 40)
        cv2.rectangle(img, (150, 150), (450, 450), (255, 255, 255), -1)
        with self.assertRaises(ValueError):
            measure_a4(img)

    def test_rejects_frame_without_dark_desk(self):
        """整幅都亮（没有深色桌面）时会被当成一张纸，靠交叉校验兜住。"""
        img = np.full((600, 600, 3), 255, np.uint8)
        with self.assertRaises(ValueError):
            measure_a4(img)

    def test_gray_input_matches_color(self):
        """文档承诺「彩色或灰度均可」，就得真的都行（曾经无条件 cvtColor 直接崩）。"""
        color = synthetic_a4()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self.assertAlmostEqual(measure_a4(gray)['ppm'], measure_a4(color)['ppm'], places=6)

    def test_sides_expose_perspective(self):
        """四边形算子要吐出逐边长度：正放矩形对边等长，梯形纸（透视）对边明显不等。"""
        rect = measure_a4(_flat_paper([[100, 150], [900, 150], [900, 716], [100, 716]]))
        self.assertEqual(rect['method'], 'quad')
        self.assertEqual(len(rect['sides']), 4)
        s = rect['sides']
        self.assertLess(abs(s[0] - s[2]) / s[2], 0.01)
        self.assertLess(abs(s[1] - s[3]) / s[3], 0.01)

        trap = measure_a4(_flat_paper([[100, 100], [900, 100], [760, 1300], [240, 1300]]), tol=99.0)
        t = trap['sides']
        self.assertGreater(abs(t[1] - t[3]) / t[3], 0.05)

    def test_rotated_paper_falls_back_to_otsu(self):
        """旋转纸光栅化后 approxPolyDP 常得到 8 顶点，四边形算子会放弃 —— 必须退回 Otsu 而不是报错。"""
        detail = measure_a4(synthetic_a4())
        self.assertEqual(detail['method'], 'otsu')
        self.assertEqual(detail['sides'], [])


class TestVerifyPpm(unittest.TestCase):
    """期间核查：只报警、不参与测量，所以它绝不能把整批处理打断。"""

    def test_skips_quietly_when_no_paper(self):
        import pipeline
        img = np.zeros((200, 200, 3), np.uint8)
        with self.assertLogs('pipeline', level='DEBUG') as cm:
            pipeline.verify_ppm(1, img)
        self.assertTrue(any('跳过期间核查' in m for m in cm.output))

    def test_warns_when_setup_drifted(self):
        import pipeline
        import config_setting
        original = config_setting.ppm
        config_setting.ppm = 10.0
        try:
            with self.assertLogs('pipeline', level='INFO') as cm:
                pipeline.verify_ppm(1, synthetic_a4())   # 合成图真值 4.0，偏差 60%
            self.assertTrue(any('装置可能漂移' in m for m in cm.output))
        finally:
            config_setting.ppm = original


if __name__ == '__main__':
    unittest.main()


