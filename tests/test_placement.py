"""试拍验收的纯逻辑必须逐条锁死：方向写反不会报错，只会照常输出并把人往错处支使。

nudge 管「往哪挪」，verdict 管「这一张能不能用」—— 后者是补上的：旧版只判位置，
放过了 ppm 已失效、量成 22.17mm 的 1元。
"""
import datetime
import tempfile
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from check_placement import (guess_coin, nudge, stale_note, verdict,  # noqa: E402
                             SCALE_GUARD_MM, BLUR_GUARD_MM)

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
        tag, bad = verdict(9.1, 0.08, 0.25)
        self.assertEqual((tag, bad), ('OK', []))

    def test_scale_failure_overrides_good_position(self):
        """现场事故原样复现：位置 9.1mm 很好，但直径离标称 2.83mm。"""
        tag, bad = verdict(9.1, -2.83, 0.25)
        self.assertEqual(tag, 'NG')
        self.assertIn('比例尺', bad[0])

    def test_unmeasurable_blur_is_not_a_failure(self):
        """对比度不够时缘宽返回 nan —— 测不出不等于不合格，不能拿它判 NG。"""
        self.assertEqual(verdict(9.1, 0.0, float('nan'))[0], 'OK')

    def test_blur_just_over_guard_fails(self):
        self.assertEqual(verdict(9.1, 0.0, BLUR_GUARD_MM + 0.01)[0], 'NG')

    def test_blur_gate_lives_in_mm_not_px(self):
        """判据必须留在毫米域。

        原先是 BLUR_GUARD_PX = 3.0。今天重标定 ppm 从 9.9749 掉到 8.6620（-13.2%），
        同一个 3px 的物理含义就从 0.301mm 变成 0.346mm —— 代码一个字没改、
        测试照样全绿、门槛却自己松了 15%。这是「数值没变但含义变了」那一类静默错值，
        本项目已经反复踩，所以把单位钉死在名字和断言里。
        """
        self.assertEqual(BLUR_GUARD_MM, 0.30)
        self.assertTrue(BLUR_GUARD_MM < 0.25 * 2)      # 不许宽到能吃掉整个直径允差
        # 同一条 0.30mm 的物理判据，在两次标定下对应的像素数不同。
        # 若门限仍写成「3px」，重标定就会让它悄悄松 15% 而无人察觉。
        self.assertLess(3.0 / 9.9749, 3.0 / 8.6620)
        self.assertEqual(verdict(9.1, 0.0, 2.8 / 9.9749)[0], 'OK')   # 2.8px @旧 ppm = 0.281mm
        self.assertEqual(verdict(9.1, 0.0, 2.8 / 8.6620)[0], 'NG')   # 2.8px @新 ppm = 0.323mm

    def test_scale_guard_is_wider_than_the_tolerance(self):
        """闸门必须比 tol 宽得多，否则正常公差波动会被当成标定失效。"""
        self.assertGreater(SCALE_GUARD_MM, 4 * 0.25)

    def test_problems_listed_scale_before_blur_before_position(self):
        """三关全挂时的排序即修复顺序：比例尺 > 清晰度 > 位置。"""
        _, bad = verdict(999.0, -5.0, 9.9)
        self.assertEqual([b[:3] for b in bad], ['比例尺', '清晰度', '位置：'])


class TestStalePhoto(unittest.TestCase):
    """照片比标定旧就必须拦。test.jpg 曾以「三关全绿」通过验收，而它差 2.1% 比例尺。"""

    CALIB = '2026-10-09 18:05:27'

    def _touch(self, folder: str, name: str, when: str) -> str:
        p = os.path.join(folder, name)
        with open(p, 'w') as f:
            f.write('x')
        ts = datetime.datetime.fromisoformat(when).timestamp()   # 本地时区，与 fromtimestamp 互逆
        os.utime(p, (ts, ts))
        return p

    def test_photo_older_than_calibration_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._touch(d, 'test.jpg', '2026-10-08 19:00:00')
            self.assertIn('照片比标定旧', stale_note(p, self.CALIB))

    def test_photo_taken_after_calibration_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._touch(d, 'a01.jpg', '2026-10-09 18:10:00')
            self.assertEqual(stale_note(p, self.CALIB), '')

    def test_missing_or_garbage_stamp_skips_check(self):
        """没标定时刻就跳过，不能因为字段缺失把每张图都判死。"""
        with tempfile.TemporaryDirectory() as d:
            p = self._touch(d, 'a01.jpg', '2026-10-08 19:00:00')
            self.assertEqual(stale_note(p, ''), '')
            self.assertEqual(stale_note(p, '昨天下午'), '')
            self.assertEqual(stale_note(os.path.join(d, '不存在.jpg'), self.CALIB), '')

    def test_same_day_minutes_apart_is_still_caught(self):
        """必须用完整时刻：标定与拍摄常只差几十分钟，日期粒度拦不住。"""
        with tempfile.TemporaryDirectory() as d:
            p = self._touch(d, 'a4_new.jpg', '2026-10-09 17:52:32')
            self.assertIn('照片比标定旧', stale_note(p, self.CALIB))


class TestGuessCoin(unittest.TestCase):
    def test_reads_denomination_from_folder(self):
        self.assertEqual(guess_coin(r'D:\shots\coins\1元\a01.jpg'), '1元')
        self.assertEqual(guess_coin(r'D:\shots\coins\5角\a01.jpg'), '5角')

    def test_defect_folder_yields_no_denomination(self):
        """defect/ 目录里没有面额，此时必须跳过直径闸门而不是拿错面额去比。"""
        self.assertEqual(guess_coin(r'D:\shots\defect\n01.jpg'), '')


if __name__ == '__main__':
    unittest.main()
