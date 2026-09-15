import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from defect_analyzer import judge_quality

class TestJudgeQuality(unittest.TestCase):
    def test_合格的硬币(self):
        result = judge_quality(25.0, 0.95, '1元')
        self.assertEqual(result['circularity'], '合格')
        self.assertEqual(result['severity'], '合格')
    def test_圆度不合格(self):
        result = judge_quality(25.0, 0.8, '1元')
        self.assertEqual(result['circularity'], '不合格')
    def test_偏大但在容差范围内(self):
        result = judge_quality(26.0, 0.95, '1元')
        self.assertEqual(result['severity'], '合格')
    def test_偏大轻度(self):
        result = judge_quality(26.3, 0.95, '1元')
        self.assertEqual(result['severity'], '轻度')
    def test_偏大中度(self):
        result = judge_quality(26.75, 0.95, '1元')
        self.assertEqual(result['severity'], '中度')
    def test_偏大严重(self):
        result = judge_quality(28.0, 0.95, '1元')
        self.assertEqual(result['severity'], '重度')
        self.assertEqual(result['direction'], '偏大')
    def test_偏小(self):
        result = judge_quality(22.0, 0.95, '1元')
        self.assertEqual(result['direction'], '偏小')
    def test_5角硬币(self):
        result = judge_quality(20.5, 0.95, '5角')
        self.assertEqual(result['severity'], '合格')
if __name__ == '__main__':
    unittest.main()
