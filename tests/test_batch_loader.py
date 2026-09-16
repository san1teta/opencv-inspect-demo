import unittest
import sys
import os
import tempfile
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'coin_detect')))
import batch_loader

def write_temp_image(folder: str, filename: str) -> str:
    path = os.path.join(folder, filename)
    cv2.imwrite(path, np.zeros((30, 30, 3), np.uint8)) 
    return path

class TestLoadImagesFromFolder(unittest.TestCase):
    def test_混合扩展名按全局字典序返回(self):
        with tempfile.TemporaryDirectory() as d:
            write_temp_image(d, 'z.jpg')
            write_temp_image(d, 'a.png')
            items = batch_loader.load_images_from_folder(d)
            self.assertEqual([x['name'] for x in items], ['a.png', 'z.jpg'])

    def test_忽略非图片扩展名与损坏文件(self):
        with tempfile.TemporaryDirectory() as d:
            write_temp_image(d, 'ok.jpg')
            with open(os.path.join(d, 'notes.txt'), 'w') as f:
                f.write('not an image')
            with open(os.path.join(d, 'broken.jpg'), 'w') as f:
                f.write('this is not jpeg data')
            items = batch_loader.load_images_from_folder(d)
            self.assertEqual([x['name'] for x in items], ['ok.jpg'])

    def test_每条记录带四个键且内容正确(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_temp_image(d, 'coin01.jpg')
            items = batch_loader.load_images_from_folder(d)
            self.assertEqual(len(items), 1)
            item = items[0]
            self.assertEqual(set(item.keys()), {'path', 'name', 'stem', 'image'})
            self.assertIsInstance(item['stem'], str)
            self.assertEqual(item['stem'], 'coin01')
            self.assertEqual(item['name'], 'coin01.jpg')
            self.assertEqual(item['path'], p)
            self.assertIsInstance(item['image'], np.ndarray)

    def test_目录不存在返回空列表(self):
        items = batch_loader.load_images_from_folder('__no_such_dir__')
        self.assertEqual(items, [])

    def test_空目录返回空列表(self):
        with tempfile.TemporaryDirectory() as d:
            items = batch_loader.load_images_from_folder(d)
            self.assertEqual(items, [])

if __name__ == '__main__':
    unittest.main()

