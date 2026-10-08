import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import exporter

FRAME = {'index': 1, 'image_name': 'a.jpg',
         'detections': [{'diameter_mm': 25.0,
                         'result': {'circularity': '合格', 'deviation': 0.0,
                                    'deficit_mm': 0.1, 'direction': '未检测',
                                    'severity': '合格', 'basis': '无'}}]}
EMPTY = {'index': 2, 'image_name': 'b.png',
          'detections': []}

class TestExporter(unittest.TestCase):
    def test_未检出的帧也要占一行(self):
        rows = exporter._export_data([FRAME, EMPTY], '1元')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][-1], '未检出')

    def test_每行宽度必须等于表头宽度(self):
        """加列时最容易漏的就是「表头改了行没改」，那样 Excel 会整体错位还不报错。"""
        rows = exporter._export_data([FRAME, EMPTY], '1元')
        for row in rows:
            self.assertEqual(len(row), len(exporter.HEADER))

    def test_缺损与依据进报表(self):
        rows = exporter._export_data([FRAME], '1元')
        self.assertEqual(rows[0][exporter.HEADER.index('缺损(mm)')], 0.1)
        self.assertEqual(rows[0][exporter.HEADER.index('判定依据')], '无')


if __name__ == '__main__':
    unittest.main()