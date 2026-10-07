import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import numpy as np

import image_io


class TestImageIo(unittest.TestCase):
    """读写都必须「读不到就抛、写不成就返回 False」，不能静默。"""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'roundtrip.png')
            img = np.zeros((30, 30, 3), np.uint8)
            img[:, :, 2] = 255
            self.assertTrue(image_io.write_image(path, img))
            back = image_io.read_image(path)
            self.assertEqual(back.shape, (30, 30, 3))
            self.assertEqual(int(back[0, 0, 2]), 255)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                image_io.read_image(os.path.join(d, 'nope.png'))

    def test_write_into_missing_dir_returns_false(self):
        """写盘失败必须返回 False 而不是抛异常或悄悄成功 —— imwrite 就是这个坑。"""
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, 'no_such_dir', 'x.png')
            self.assertFalse(image_io.write_image(bad, np.zeros((10, 10, 3), np.uint8)))


if __name__ == '__main__':
    unittest.main()
