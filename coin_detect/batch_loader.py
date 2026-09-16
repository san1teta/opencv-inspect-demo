import os
import cv2
import glob
import logging
from typing import Any

logger = logging.getLogger(__name__)

def load_images_from_folder(folder_path: str) -> list[dict[str, Any]]:
    if not os.path.isdir(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")   
        return []

    img_specs = ['*.jpg', '*.png', '*.bmp']
    
    paths = []
    for spec in img_specs:
        paths.extend(glob.glob(os.path.join(folder_path, spec)))
    paths.sort()
    items = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            logger.warning(f"无法读取图像: {p}")
            continue
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        items.append({
            'path': p,
            'name': name,
            'stem': stem,
            'image': img
        })
    return items