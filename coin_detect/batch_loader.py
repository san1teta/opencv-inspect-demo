import os
import cv2
import glob

def load_images_from_folder(folder_path):
    paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    images = []
    for p in paths:
        img = cv2.imread(p)
        images.append(img)
    return images