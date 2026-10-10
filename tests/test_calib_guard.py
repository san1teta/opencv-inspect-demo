"""「照片比标定旧」这道闸门必须同时管住验收工具和批量流水线。

回归的是**重标定这个动作本身**的风险：装置一动 ppm 就换，而旧照片还躺在样本目录里。
batch 模式会用新 ppm 去量旧照片，得到一个看着完全正常的直径，混进均值和标准差 ——
全程不报错，检出率还是 100%。2026-10-09 已为同一类问题踩过一次（尺寸闸门只长在工具里），
这次趁重标定之前把判据统一到 calib_guard，两边共用。
"""
import contextlib
import datetime
import io
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'coin_detect'))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

import batch_loader  # noqa: E402
import config_setting  # noqa: E402
import report_metrics  # noqa: E402
from calib_guard import photo_vs_calibration, stale_paths  # noqa: E402

STAMP = '2026-10-10 09:00:00'


def _touch(path: str, when: str) -> str:
    ts = datetime.datetime.fromisoformat(when).timestamp()   # 本地时区，与 fromtimestamp 互逆
    os.utime(path, (ts, ts))
    return path


def _jpg(path: str) -> str:
    """造一张真能读出来的小图。batch_loader 会真的解码，空文件会被 except 掉、测不到警告。"""
    ok, buf = cv2.imencode('.jpg', np.full((60, 60, 3), 128, np.uint8))
    assert ok
    buf.tofile(path)
    return path


class TestPhotoVsCalibration(unittest.TestCase):
    def test_older_than_calibration_is_returned(self):
        with tempfile.TemporaryDirectory() as d:
            p = _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-09 18:54:00')
            got = photo_vs_calibration(p, STAMP)
            self.assertIsNotNone(got)
            assert got is not None      # 给类型检查器收窄，下面才能下标
            self.assertEqual(got[0].strftime('%Y-%m-%d %H:%M'), '2026-10-09 18:54')

    def test_taken_after_calibration_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-10 09:30:00')
            self.assertIsNone(photo_vs_calibration(p, STAMP))

    def test_empty_stamp_never_flags(self):
        """还没标定（calib_stamp 为空）不能把每张图都判成过期。"""
        with tempfile.TemporaryDirectory() as d:
            p = _touch(_jpg(os.path.join(d, 'a01.jpg')), '2020-01-01 00:00:00')
            self.assertIsNone(photo_vs_calibration(p, ''))

    def test_garbage_stamp_and_missing_file_never_flags(self):
        """宁可漏判，不可冤枉：信息坏了就跳过，不能因为字段异常把整批图拦死。"""
        with tempfile.TemporaryDirectory() as d:
            p = _jpg(os.path.join(d, 'a01.jpg'))
            self.assertIsNone(photo_vs_calibration(p, '昨天下午'))
            self.assertIsNone(photo_vs_calibration(os.path.join(d, '不存在.jpg'), STAMP))

    def test_same_day_minutes_apart_is_caught(self):
        """标定与拍摄常只差几十分钟，日期粒度拦不住 —— 必须用完整时刻。"""
        with tempfile.TemporaryDirectory() as d:
            p = _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-10 08:59:00')
            self.assertIsNotNone(photo_vs_calibration(p, STAMP))

    def test_stale_paths_keeps_order_and_subset(self):
        with tempfile.TemporaryDirectory() as d:
            old = _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-09 18:00:00')
            new = _touch(_jpg(os.path.join(d, 'a02.jpg')), '2026-10-10 10:00:00')
            self.assertEqual(stale_paths([new, old], STAMP), [old])



class TestBatchLoaderWarns(unittest.TestCase):
    """批量模式必须自己发现混批 —— 不能只在验收工具里拦。"""

    def setUp(self):
        self._orig = config_setting.calib_stamp
        self.addCleanup(setattr, config_setting, 'calib_stamp', self._orig)
        config_setting.calib_stamp = STAMP

    def test_stale_photo_in_folder_triggers_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-09 18:54:00')
            _touch(_jpg(os.path.join(d, 'a02.jpg')), '2026-10-10 09:30:00')
            with self.assertLogs('batch_loader', level='WARNING') as cap:
                items = batch_loader.load_images_from_folder(d)
            self.assertEqual(len(items), 2, '只警告，不替人丢弃照片')
            self.assertIn('比标定', cap.output[0])
            self.assertIn('a01.jpg', cap.output[0])
            self.assertNotIn('a02.jpg', cap.output[0])

    def test_fresh_folder_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-10 09:30:00')
            with self.assertNoLogs('batch_loader', level='WARNING'):
                self.assertEqual(len(batch_loader.load_images_from_folder(d)), 1)

    def test_no_calibration_never_warns(self):
        """没标定时刻时批量模式照常跑，不拿「过期」当理由刷屏。"""
        config_setting.calib_stamp = ''
        with tempfile.TemporaryDirectory() as d:
            _touch(_jpg(os.path.join(d, 'a01.jpg')), '2020-01-01 00:00:00')
            with self.assertNoLogs('batch_loader', level='WARNING'):
                self.assertEqual(len(batch_loader.load_images_from_folder(d)), 1)


class TestReportMetricsWarns(unittest.TestCase):
    """报表工具同样要报 —— 它才是那些数字的出口。"""

    def setUp(self):
        self._orig = config_setting.calib_stamp
        self.addCleanup(setattr, config_setting, 'calib_stamp', self._orig)
        config_setting.calib_stamp = STAMP

    def test_stale_photo_prints_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-09 18:54:00')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats = report_metrics._stats(d, '1元')
            self.assertIsNotNone(stats)
            self.assertIn('比标定', buf.getvalue())
            self.assertIn('a01.jpg', buf.getvalue())

    def test_fresh_folder_prints_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(_jpg(os.path.join(d, 'a01.jpg')), '2026-10-10 09:30:00')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report_metrics._stats(d, '1元')
            self.assertNotIn('比标定', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
