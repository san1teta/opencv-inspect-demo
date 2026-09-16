import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

import exporter

FRAME = {'index': 1, 'image_name': 'a.jpg',
         'detections': [{'diameter_mm': 25.0,
                         'result': {'circularity': '合格', 'deviation': 0.0,
                                    'direction': '未检测', 'severity': '合格'}}]}
EMPTY = {'index': 2, 'image_name': 'b.png',
          'detections': []}

class TestExporter(unittest.TestCase):
    def test_未检出的帧也要占一行(self):
        rows = exporter._export_data([FRAME, EMPTY], '1元')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][7], '未检出')

if __name__ == '__main__':
    unittest.main()