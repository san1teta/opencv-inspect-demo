"""放位方向必须逐向锁死：方向写反不会报错，只会照常输出并把人往错处支使。"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from check_placement import nudge  # noqa: E402

PPM = 9.9749
T = (1000.0, 1000.0)   # 纸心


def dirs(cx, cy):
    return [d for d, _ in nudge(T[0], T[1], cx, cy, PPM)[2]]


class TestNudge(unittest.TestCase):
    def test_below_target_moves_up(self):
        """图像 y 轴向下：硬币在纸心下方 => 要往「上」挪。这条最容易写反。"""
        self.assertEqual(dirs(1000, 1000 + 69 * PPM), ['上'])

    def test_above_target_moves_down(self):
        self.assertEqual(dirs(1000, 1000 - 30 * PPM), ['下'])

    def test_right_of_target_moves_left(self):
        self.assertEqual(dirs(1000 + 20 * PPM, 1000), ['左'])

    def test_left_of_target_moves_right(self):
        self.assertEqual(dirs(1000 - 20 * PPM, 1000), ['右'])

    def test_both_axes_listed_vertical_first(self):
        self.assertEqual(dirs(1000 + 40 * PPM, 1000 - 30 * PPM), ['下', '左'])

    def test_amount_is_in_mm_not_px(self):
        parts = nudge(T[0], T[1], 1000, 1000 + 69 * PPM, PPM)[2]
        self.assertAlmostEqual(parts[0][1], 69.0, delta=1.0)

    def test_sub_2mm_is_ignored(self):
        """2mm 以内的抖动不值得支使人去挪，否则每次都说要挪，人就学会忽略工具了。"""
        self.assertEqual(nudge(T[0], T[1], 1003.0, 1002.0, PPM)[2], [])


if __name__ == '__main__':
    unittest.main()
