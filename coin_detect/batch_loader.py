import os
import cv2
import glob

def load_images_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"文件夹不存在: {folder_path}")   
        return []

    img_specs = ['*.jpg', '*.png', '*.bmp']
    
    paths = []
    for spec in img_specs:
        paths.extend(glob.glob(os.path.join(folder_path, spec)))

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"无法读取图像: {p}")
            continue
        else:
            images.append(img)
    return images