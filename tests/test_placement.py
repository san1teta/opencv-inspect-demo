"""试拍验收的纯逻辑必须逐条锁死：方向写反不会报错，只会照常输出并把人往错处支使。

nudge 管「往哪挪」，verdict 管「这一张能不能用」—— 后者是补上的：旧版只判位置，
放过了 ppm 已失效、量成 22.17mm 的 1元。
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from check_placement import guess_coin, nudge, verdict, SCALE_GUARD_MM, BLUR_GUARD_PX  # noqa: E402

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


class TestVerdict(unittest.TestCase):
    """三关必须同过。旧版只校验位置，放过了量成 22.17mm 的 1元 —— 它校验了唯一没坏的那一项。"""

    def test_all_three_clear_is_ok(self):
        tag, bad = verdict(9.1, 0.08, 2.5)
        self.assertEqual((tag, bad), ('OK', []))

    def test_scale_failure_overrides_good_position(self):
        """现场事故原样复现：位置 9.1mm 很好，但直径离标称 2.83mm。"""
        tag, bad = verdict(9.1, -2.83, 2.5)
        self.assertEqual(tag, 'NG')
        self.assertIn('比例尺', bad[0])

    def test_unmeasurable_blur_is_not_a_failure(self):
        """对比度不够时缘宽返回 nan —— 测不出不等于不合格，不能拿它判 NG。"""
        self.assertEqual(verdict(9.1, 0.0, float('nan'))[0], 'OK')

    def test_blur_just_over_guard_fails(self):
        self.assertEqual(verdict(9.1, 0.0, BLUR_GUARD_PX + 0.1)[0], 'NG')

    def test_scale_guard_is_wider_than_the_tolerance(self):
        """闸门必须比 tol 宽得多，否则正常公差波动会被当成标定失效。"""
        self.assertGreater(SCALE_GUARD_MM, 4 * 0.25)

    def test_problems_listed_scale_before_blur_before_position(self):
        """三关全挂时的排序即修复顺序：比例尺 > 清晰度 > 位置。"""
        _, bad = verdict(999.0, -5.0, 9.9)
        self.assertEqual([b[:3] for b in bad], ['比例尺', '清晰度', '位置：'])


class TestGuessCoin(unittest.TestCase):
    def test_reads_denomination_from_folder(self):
        self.assertEqual(guess_coin(r'D:\shots\coins\1元\a01.jpg'), '1元')
        self.assertEqual(guess_coin(r'D:\shots\coins\5角\a01.jpg'), '5角')

    def test_defect_folder_yields_no_denomination(self):
        """defect/ 目录里没有面额，此时必须跳过直径闸门而不是拿错面额去比。"""
        self.assertEqual(guess_coin(r'D:\shots\defect\n01.jpg'), '')


if __name__ == '__main__':
    unittest.main()
