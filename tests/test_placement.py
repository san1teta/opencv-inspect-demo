"""试拍验收的纯逻辑必须逐条锁死：方向写反不会报错，只会照常输出并把人往错处支使。

nudge 管「往哪挪」，verdict 管「这一张能不能用」—— 后者是补上的：旧版只判位置，
放过了 ppm 已失效、量成 22.17mm 的 1元。
"""
import datetime
import tempfile
import unittest
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

from check_placement import (guess_coin, nudge, stale_note, verdict,  # noqa: E402
                             explain_no_coin, edge_blur_px, pick_coins,
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


class TestExplainNoCoin(unittest.TestCase):
    """检不到币时必须说破原因，光一句「没找到」在现场等于没说。"""

    def test_blank_image_reports_no_contour(self):
        import numpy as np
        lines = explain_no_coin(np.zeros((400, 300, 3), dtype=np.uint8))
        self.assertTrue(any('一个轮廓都没有' in x for x in lines), lines)

    def test_coin_on_paper_names_the_a4_sheet(self):
        """RETR_EXTERNAL 下币是嵌套轮廓会被整个丢掉，只剩纸 —— 这个坑必须点名。"""
        import cv2
        import numpy as np
        import config_setting
        img = np.zeros((4096, 3072, 3), dtype=np.uint8)
        cx, cy = config_setting.calib_center
        hw = 297 * config_setting.ppm / 2      # A4 长边一半（像素）
        hh = 210 * config_setting.ppm / 2
        cv2.rectangle(img, (int(cx - hw), int(cy - hh)),
                      (int(cx + hw), int(cy + hh)), (230, 230, 230), -1)
        cv2.circle(img, (int(cx), int(cy)), int(12.5 * config_setting.ppm / 2),
                   (40, 40, 40), -1)
        joined = '\n'.join(explain_no_coin(img))
        self.assertIn('A4', joined, joined)
        self.assertIn('嵌套', joined, joined)


class TestEdgeBlurMeasurement(unittest.TestCase):
    """edge_blur_px 此前**零覆盖** —— 上面 4 条 blur 测试全部只给 verdict() 喂手写数字。

    后果：第一张真照片 a01.jpg 上它返回 nan，verdict 按设计「nan 不算 NG」，
    于是清晰度闸门整条静默不执行、输出连「缘宽」字段都不打印，而测试全绿。
    判据写对了、测量返回 nan，一样是废的 —— 测量函数必须自己被测。
    """

    C = 200          # 圆心（图像坐标，y 向下不影响环形平均）
    R = 100.0        # 半径；环形窗口 band=20 → 采样 r∈[80,120]，台阶落在正中

    @staticmethod
    def _layered(levels: list[tuple[float, int]]) -> np.ndarray:
        """按半径分层的灰度图。levels 从外到内给 (半径, 灰度)，依次画实心圆覆盖。

        单通道图用标量灰度是对的 —— 只有 3 通道 BGR 图才必须给三元组。
        """
        img = np.zeros((401, 401), np.uint8)
        for radius, val in levels:
            cv2.circle(img, (TestEdgeBlurMeasurement.C, TestEdgeBlurMeasurement.C),
                       int(round(radius)), int(val), -1)
        return img

    def _blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        k = int(6 * sigma) // 2 * 2 + 1
        return cv2.GaussianBlur(img, (k, k), sigma)

    def test_sharp_high_contrast_edge_is_measurable(self):
        """最基础的一条：干净的高对比锐边必须测出数，不能返回 nan。

        这条正是 a01.jpg 上塌掉的那一支 —— nan 被当成「测不到」悄悄放过。
        """
        px = edge_blur_px(self._layered([(200, 20), (self.R, 200)]), self.C, self.C, self.R)
        self.assertFalse(np.isnan(px), '锐利高对比边缘不该返回 nan')
        self.assertLess(px, 2.5, '实心圆边缘只有像素量化宽度，不该接近门限')

    def test_measurement_floor_sits_below_the_gate(self):
        """量化的「绝对锐利」读数是 1.6px 底噪，必须明显低于门限，否则闸门名存实亡。

        BLUR_GUARD_MM=0.30 在 ppm=8.662 下 = 2.60px，底噪 1.63px → 真正可用的余量只有
        ~1px。这条断言把这个余量钉住：将来改门限 / 换 ppm 时它会提醒还剩多少空间，
        而不是让人以为 0.30mm 是从零起算的。
        """
        floor_px = edge_blur_px(self._layered([(200, 20), (self.R, 200)]), self.C, self.C, self.R)
        self.assertGreater(floor_px, 0.0)
        gate_px_at_new_ppm = BLUR_GUARD_MM * 8.6620
        self.assertGreater(gate_px_at_new_ppm - floor_px, 0.5,
                           f'门限距底噪仅 {gate_px_at_new_ppm - floor_px:.2f}px，判据分不开糊与锐')

    def test_measured_width_matches_gaussian_rise_width(self):
        """量化正确性：高斯模糊台阶的 10~90% 过渡宽度理论值 = 2×1.2816σ = 2.563σ。

        只测「能算出个数」不够，得证明那个数是边缘宽度而不是别的什么东西。
        """
        for sigma in (1.0, 2.0):
            img = self._layered([(200, 20), (self.R, 200)])
            px = edge_blur_px(self._blur(img, sigma), self.C, self.C, self.R)
            self.assertFalse(np.isnan(px), f'sigma={sigma} 测不出')
            self.assertAlmostEqual(px / (2.5632 * sigma), 1.0, delta=0.2,
                                   msg=f'sigma={sigma} 实测 {px:.2f}px 偏离理论 {2.5632 * sigma:.2f}px')

    def test_blur_monotonic_in_sigma(self):
        """糊得越厉害数值必须越大 —— 否则这个量根本不能当清晰度用。"""
        img = self._layered([(200, 20), (self.R, 200)])
        vals = [edge_blur_px(self._blur(img, s), self.C, self.C, self.R) for s in (0.5, 1.5, 3.0)]
        self.assertTrue(all(not np.isnan(v) for v in vals), f'有 nan: {vals}')
        self.assertEqual(vals, sorted(vals), f'非单调: {vals}')

    def test_dark_coin_on_bright_background(self):
        """反相（暗币亮底）走 top<bot 翻转分支，不能翻车成 nan 或负数。"""
        px = edge_blur_px(self._layered([(200, 200), (self.R, 20)]), self.C, self.C, self.R)
        self.assertFalse(np.isnan(px))
        self.assertGreater(px, 0.0)

    def test_bright_annulus_inside_rim_still_finds_crossing(self):
        """回归 a01.jpg 的真实故障：币缘内侧一圈亮带把 90% 交点推到最陡点**右侧**。

        剖面 r<88:70 / 88~96:120 / 96~100:75 / >100:30。最陡下落点在 r=96，
        而 90% 电平 66 唯一的交点在 r=100 —— 在 j 的右边。
        旧实现写死「往左找交点」，一路走到 r=80 全在电平之上 → 返回 nan → 闸门不执行。
        """
        img = self._layered([(200, 30), (self.R, 75), (96, 120), (88, 70)])
        px = edge_blur_px(img, self.C, self.C, self.R)
        self.assertFalse(np.isnan(px), '90% 交点落在最陡点右侧时旧实现返回 nan')
        self.assertLess(px, 2.0, f'分层图边缘本该很锐，实测 {px:.2f}px')

    def test_low_contrast_returns_nan_on_purpose(self):
        """对比度不足要老实返回 nan，而不是硬凑一个宽度 —— nan 由 verdict 显式放行。"""
        px = edge_blur_px(self._layered([(200, 100), (self.R, 105)]), self.C, self.C, self.R)
        self.assertTrue(np.isnan(px))

    def test_real_photo_blur_gate_is_actually_exercised(self):
        """闸门必须真的在跑：合成一条糊边，确认 verdict 拿到的是有限值并判 NG。

        拦的就是本次那个失效模式 —— 测量返回 nan，判据「正确地」不报错，全链路绿灯。
        """
        img = self._blur(self._layered([(200, 20), (self.R, 200)]), 3.0)
        px = edge_blur_px(img, self.C, self.C, self.R)
        self.assertFalse(np.isnan(px))
        ppm = 8.6620
        tag, bad = verdict(9.1, 0.0, px / ppm)
        self.assertEqual(tag, 'NG', f'σ=3 的糊边（{px / ppm:.3f}mm）竟然放行')
        self.assertIn('清晰度', bad[0])


class TestPickCoins(unittest.TestCase):
    """检出闸门必须与 pipeline 同源：用直径窗口，不用圆度。

    现场代价说清楚：拍缺陷样（贴胶带咬掉一块）时圆度会掉。旧实现在这里就用
    `circularity >= 0.9` 筛，工具会报「没找到可信的圆形目标」，于是人去重拍一张
    本来完全正常的照片 —— 而真正该报的是「这枚币变形，重度不合格」。
    """

    PPM = 8.6620
    SPEC = {'diameter': 22.25}          # 2019 版 1元

    @staticmethod
    def _f(dia_mm: float, circ: float = 0.98) -> dict:
        return {'diameter_pixels': dia_mm * TestPickCoins.PPM, 'circularity': circ}

    def test_deformed_coin_survives(self):
        """圆度 0.75 的变形币必须留下，交给下游如实判不合格。"""
        self.assertEqual(len(pick_coins([self._f(22.25, 0.75)], self.SPEC, self.PPM)), 1)

    def test_background_blobs_dropped(self):
        """昨天那 50 个 1.3~4.8mm 背景团，一个都不许进报表。"""
        blobs = [self._f(d, 0.95) for d in (1.3, 2.0, 3.1, 4.8)]
        self.assertEqual(pick_coins(blobs, self.SPEC, self.PPM), [])

    def test_wrong_denomination_dropped(self):
        """1角(19mm) 混进 1元 批次：它圆度 0.983 极高，只有尺寸窗口能剔掉。"""
        self.assertEqual(pick_coins([self._f(19.0, 0.983)], self.SPEC, self.PPM), [])

    def test_gate_is_the_detection_window_not_the_tolerance(self):
        """窗口是 ±diameter_gate_mm（这是不是枚币），不是 ±0.25mm（这枚币合不合格）。

        两者混用会把尺寸超差的币当背景悄悄丢掉，而不是报成重度不合格。
        """
        self.assertEqual(len(pick_coins([self._f(22.25 - 1.5)], self.SPEC, self.PPM)), 1)
        self.assertEqual(pick_coins([self._f(22.25 - 2.5)], self.SPEC, self.PPM), [])

    def test_no_denomination_falls_back_to_circularity(self):
        """认不出面额时退回圆度筛是**降级**，不是等价方案 —— 行为钉在这里，
        将来换成别的方案这条测试会先响。"""
        self.assertEqual(len(pick_coins([self._f(22.25, 0.95)], None, self.PPM)), 1)
        self.assertEqual(pick_coins([self._f(22.25, 0.50)], None, self.PPM), [])


if __name__ == '__main__':
    unittest.main()

