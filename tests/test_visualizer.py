"""visualizer 的两个静默失败点：中文标签会变乱码字形，分级代号漏一个就 KeyError。"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import numpy as np

import visualizer

BOX = np.array([[[10, 10]], [[50, 10]], [[50, 60]], [[10, 60]]], np.int32)


def _feat(severity, direction, circularity='合格'):
    return {'contour': BOX, 'diameter_mm': 25.08,
            'result': {'circularity': circularity, 'severity': severity,
                       'direction': direction, 'deviation': 0.08}}


class TestVisualizer(unittest.TestCase):
    def test_all_label_text_is_ascii(self):
        """cv2.putText 不认中文，也不报错，只在图上画出 mmmnn 这种垃圾字形。"""
        texts = (list(visualizer.CODE.values()) + list(visualizer.DIRECTION.values())
                 + [visualizer.SHAPE_NG])
        for t in texts:
            self.assertTrue(t.isascii(), f'{t!r} 画到图上会变乱码')

    def test_every_grade_has_an_ascii_code(self):
        """配色表和代号表必须同键，否则画到某一档就 KeyError。"""
        self.assertEqual(set(visualizer.color_map), set(visualizer.CODE))

    def test_draws_something_for_each_verdict(self):
        img = np.zeros((400, 600, 3), np.uint8)
        cases = [('合格', '未检测'), ('轻度', '偏大'), ('中度', '偏大'), ('重度', '偏小'),
                 ('合格', '未检测')]
        for severity, direction in cases:
            with self.subTest(severity=severity):
                out = visualizer.draw_result(img, [_feat(severity, direction)])
                self.assertFalse((out == img).all(), '什么都没画出来')

    def test_shape_ng_when_circle_gate_fails(self):
        img = np.zeros((400, 600, 3), np.uint8)
        out = visualizer.draw_result(img, [_feat('合格', '未检测', circularity='不合格')])
        self.assertTrue((out == (0, 0, 255)).any(), '形状不合格应该标红')


if __name__ == '__main__':
    unittest.main()
