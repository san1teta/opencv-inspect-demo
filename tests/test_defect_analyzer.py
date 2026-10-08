import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from defect_analyzer import judge_quality

OK_DEFICIT = 0.10   # 完整圆的径向凹陷噪声底（实测值）


def _q(diameter, deficit=OK_DEFICIT, circularity=0.95, coin='1元'):
    return judge_quality(diameter, deficit, circularity, coin)


class TestSizePath(unittest.TestCase):
    """尺寸路：超出量 = ||直径 - 标称| - 允差(0.25)|，再按 0.3 / 0.6 分档。"""

    def test_合格(self):
        r = _q(25.0)
        self.assertEqual((r['circularity'], r['severity'], r['basis']), ('合格', '合格', '无'))

    def test_允差内不算超差(self):
        self.assertEqual(_q(25.2)['severity'], '合格')

    def test_偏大轻度(self):
        r = _q(25.3)
        self.assertEqual((r['severity'], r['direction'], r['basis']), ('轻度', '偏大', '尺寸偏大'))

    def test_偏大中度(self):
        self.assertEqual(_q(25.6)['severity'], '中度')

    def test_偏大重度(self):
        self.assertEqual(_q(25.9)['severity'], '重度')

    def test_偏小(self):
        r = _q(24.5)
        self.assertEqual((r['severity'], r['direction']), ('轻度', '偏小'))

    def test_5角用5角的标称(self):
        self.assertEqual(_q(20.5, coin='5角')['severity'], '合格')
        self.assertEqual(_q(21.5, coin='5角')['severity'], '重度')


class TestEdgePath(unittest.TestCase):
    """边缘路：径向最大凹陷按 0.4 / 0.8 / 1.5mm 分档。"""

    def test_噪声底不误报(self):
        self.assertEqual(_q(25.0, deficit=0.10)['severity'], '合格')

    def test_三档依次触发(self):
        self.assertEqual(_q(25.0, deficit=0.50)['severity'], '轻度')
        self.assertEqual(_q(25.0, deficit=1.00)['severity'], '中度')
        self.assertEqual(_q(25.0, deficit=2.00)['severity'], '重度')

    def test_缺口直径看不出来但凹陷能看出来(self):
        """这条是换判据的全部理由：1mm 深的缺口只让等效直径动 0.035mm（尺寸路必然漏）。"""
        r = _q(25.035, deficit=1.001)
        self.assertEqual(r['severity'], '中度')
        self.assertEqual(r['basis'], '刀口1.00mm')
        self.assertEqual(r['direction'], '缺损')


class TestCombination(unittest.TestCase):
    def test_取更差的一档(self):
        """尺寸轻度 + 刀口重度 = 重度，不能相加也不能取先到的一条。"""
        r = _q(25.3, deficit=2.0)
        self.assertEqual(r['severity'], '重度')
        self.assertEqual(r['basis'], '刀口2.00mm')

    def test_同档两路都写进依据(self):
        r = _q(25.9, deficit=2.0)
        self.assertEqual(r['severity'], '重度')
        self.assertEqual(r['basis'], '尺寸偏大+刀口2.00mm')

    def test_形状不合格早退(self):
        r = _q(25.0, deficit=3.0, circularity=0.5)
        self.assertEqual(r['circularity'], '不合格')
        self.assertEqual(r['severity'], '未检测')

    def test_缺损量进报表(self):
        self.assertEqual(_q(25.0, deficit=1.234)['deficit_mm'], 1.234)


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
