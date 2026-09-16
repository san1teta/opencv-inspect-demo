import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))

from calibration import pixels_to_mm

class TestPixelsToMm(unittest.TestCase):
    def test_pixels_to_mm(self):
        self.assertEqual(pixels_to_mm(100, 10.0), 10.0)
        
if __name__ == '__main__':
    unittest.main()

