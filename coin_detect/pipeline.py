import preprocess
import detector
import config_setting
import defect_analyzer
import visualizer
import calibration
from batch_loader import load_images_from_folder

import logging
import numpy as np
from typing import Any
import os
import cv2
import sys

logger = logging.getLogger(__name__)

def process_single_frame(frame: np.ndarray, coin_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """单帧定义"""
    gray_img = preprocess.gray(frame)
    blur_img = preprocess.gaussian_blur(gray_img, config_setting.gaussian_kernel)
    edge_img = preprocess.canny(blur_img, config_setting.canny_low, config_setting.canny_high)
    contours = detector.find_contours(edge_img, config_setting.min_area)
    all_features = []
    for cnt in contours:
        features = detector.calculate_features(cnt)
        diameter_mm = calibration.pixels_to_mm(features['diameter_pixels'], config_setting.ppm)
        result = defect_analyzer.judge_quality(diameter_mm, features['circularity'], coin_name)
        all_features.append({
            'contour': cnt,
            'diameter_mm': diameter_mm,
            'result': result
        })
        logger.debug(f"{coin_name}的结果是{result}")
        
    detect_visualize = visualizer.draw_result(frame,all_features)

    return detect_visualize, all_features

def run_batch(input_path: str, output_path: str, coin_name: str) -> list[list[dict[str,Any]]]:
    imgs = load_images_from_folder(input_path)
    if not imgs:
        logger.error("没有找到任何图片")
        sys.exit(1)
    os.makedirs(output_path, exist_ok=True)
    all_results = []
    for i, img in enumerate(imgs):
        logger.info(f"开始处理第{i+1}张图片...") 
        result_img, all_features = process_single_frame(img, coin_name)
        cv2.imwrite(os.path.join(output_path, f"result_{i}.jpg"), result_img) 
        all_results.append(all_features) 
        logger.info(f"处理完成第{i+1}张图片")
    return all_results