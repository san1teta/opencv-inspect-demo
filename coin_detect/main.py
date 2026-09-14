import argparse
import preprocess
import detector
import defect_analyzer
import visualizer
import exporter

import colibration
import config_setting

from batch_loader import load_images_from_folder
from camera_capture import start_camera_loop
import cv2
import os
import sys
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

cal = colibration.Calibrator(config_setting.ppm)

def process_single_frame(frame: np.ndarray, coin_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """单帧定义"""
    gray_img = preprocess.gray(frame)
    blur_img = preprocess.gaussian_blur(gray_img, config_setting.gaussion_kernel)
    edge_img = preprocess.canny(blur_img, config_setting.canny_low, config_setting.canny_high)
    contours = detector.find_contours(edge_img, config_setting.min_area)
    all_features = []
    for cnt in contours:
        features = detector.calculate_features(cnt)
        diameter_mm = cal.pixels_to_mm(features['diameter_pixels'])
        result = defect_analyzer.judge_quality(diameter_mm, features['circularity'], coin_name)
        all_features.append({
            'contour': cnt,
            'diameter_mm': diameter_mm,
            'result': result
        })
        logger.debug(f"{coin_name}的结果是{result}")
        
    detect_visualize = visualizer.draw_result(frame,all_features)

    return detect_visualize, all_features


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s [%(levelname)s] %(name)s - %(message)'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "camera"])
    parser.add_argument("--coin", type=str, required=True, choices=["1元", "5角", "1角"])
    parser.add_argument("--input", "-i", type=str, help="待检测图片文件夹")
    parser.add_argument("--output", "-o", type=str, help="输出图片路径")
    args = parser.parse_args()

    if args.mode == "batch":
        if not args.input:
            sys.exit("请输入图片路径")
        output_path = args.output or os.path.join(args.input, "result")
        logger.info("进入批量模式...")
        imgs = load_images_from_folder(args.input)
        if not imgs:
            sys.exit("没有找到任何图片")
        os.makedirs(output_path, exist_ok=True)
        all_results = []
        for i, img in enumerate(imgs):
            logger.info(f"开始处理第{i+1}张图片...") 
            result_img, all_features = process_single_frame(img, args.coin)
            cv2.imwrite(os.path.join(output_path, f"result_{i}.jpg"), result_img) 
            all_results.append(all_features) 
            logger.info(f"处理完成第{i+1}张图片")
        exporter.export_to_csv(all_results, os.path.join(output_path, "coin_detect_report.csv"),args.coin)
        exporter.export_to_excel(all_results, os.path.join(output_path, "coin_detect_report.xlsx"), args.coin)
    else:
        logger.info("进入实时模式...")
        start_camera_loop(lambda frame: process_single_frame(frame, args.coin)[0])
