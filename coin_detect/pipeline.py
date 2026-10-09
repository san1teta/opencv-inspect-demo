import preprocess
import detector
import config_setting
import defect_analyzer
import visualizer
import calibration
import image_io
from batch_loader import load_images_from_folder

import logging
import numpy as np
from typing import Any
import os
import sys

logger = logging.getLogger(__name__)

def _drop_background(pairs: list[tuple[np.ndarray, dict[str, Any]]], coin_name: str) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """丢掉尺寸根本不像这个面额的轮廓 —— 它们是背景，不是币。

    2026-10-09 第一张实拍才暴露出这一步缺失：鼠标垫织纹 + 画面下方一个失焦的桶，
    batch 模式检出 51 个「币」（1.3~4.8mm），每个在 CSV 里占一行，均值/σ/检出率全废。
    当时 check_placement 对同一张图只报 1 个目标，因为它自带 circularity 过滤 ——
    闸门只长在工具里、没长在流水线上，而报表走的是流水线。

    circularity 不能拿来当这里的闸门：圆度低是「这枚币变形了」，属于质量结论，
    必须留在报表里判不合格；拿它当检出条件就会把最该报警的币悄悄吞掉。
    """
    spec = config_setting.coin_specs.get(coin_name)
    if spec is None:
        return pairs
    lo, hi = detector.diameter_window(spec['diameter'], config_setting.ppm,
                                      config_setting.diameter_gate_mm)
    kept = [(c, f) for c, f in pairs if lo <= f['diameter_pixels'] <= hi]
    if pairs and not kept:
        # 全军覆没最可能是标定/距离出了问题，而不是「画面里没有币」，所以把最大候选的
        # 实际直径写进日志，别只留一句「未检出」让人去猜是哪一类错。
        big = max((f for _, f in pairs), key=lambda f: f['area'])
        logger.warning(f"{coin_name}：{len(pairs)} 个轮廓全部落在标称 "
                       f"{spec['diameter']}±{config_setting.diameter_gate_mm}mm 之外"
                       f"（最大者等效直径 {big['diameter_pixels'] / config_setting.ppm:.2f}mm、"
                       f"圆度 {big['circularity']:.3f}）—— 先查相机距离与 ppm，再确认币有没有入画")
    return kept


def process_single_frame(frame: np.ndarray, coin_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """单帧定义"""
    gray_img = preprocess.gray(frame)
    blur_img = preprocess.gaussian_blur(gray_img, config_setting.gaussian_kernel)
    edge_img = preprocess.canny(blur_img, config_setting.canny_low, config_setting.canny_high)
    contours = detector.find_contours(edge_img, config_setting.min_area)
    pairs = _drop_background([(cnt, detector.calculate_features(cnt)) for cnt in contours],
                             coin_name)
    all_features = []
    for cnt, features in pairs:
        diameter_mm = calibration.pixels_to_mm(features['diameter_pixels'], config_setting.ppm)
        deficit_mm = calibration.pixels_to_mm(features['deficit_pixels'], config_setting.ppm)
        result = defect_analyzer.judge_quality(diameter_mm, deficit_mm,
                                               features['circularity'], coin_name)
        all_features.append({
            'contour': cnt,
            'diameter_mm': diameter_mm,
            # 数值圆度单独留一份：judge_quality 把 result['circularity'] 覆成了
            # 「合格/不合格」，光看报表拿不到分布，而 min_circularity=0.90 这条临时
            # 门限正是靠真样本的分布来定的。
            'circularity': features['circularity'],
            'deficit_point': features['deficit_point'],
            'result': result
        })
        logger.debug(f"{coin_name}的结果是{result}")
        
    detect_visualize = visualizer.draw_result(frame,all_features)

    return detect_visualize, all_features

def verify_ppm(index: int, frame: np.ndarray) -> None:
    """期间核查：量本图里那张 A4 的 ppm，与锁定的标定值比对。

    只报警、不参与计算 —— 直径永远用 config_setting.ppm，保持「标定一次不改动」的工业口径。
    图里没有可信的 A4（被遮挡/没入画）就静默跳过，不算错误。
    """
    try:
        ppm_now = calibration.measure_a4(frame)['ppm']
    except ValueError:
        logger.debug(f"第{index}张图里没有可信的 A4 纸，跳过期间核查")
        return
    drift = abs(ppm_now - config_setting.ppm) / config_setting.ppm
    if drift > config_setting.ppm_check_tol:
        logger.warning(f"第{index}张图期间核查：本图实测 {ppm_now:.3f} ppm，"
                       f"与标定值 {config_setting.ppm:.3f} 差 {drift * 100:.1f}%，装置可能漂移")
    else:
        logger.info(f"第{index}张图期间核查通过：本图实测 {ppm_now:.3f} ppm，偏差 {drift * 100:.1f}%")


def run_batch(input_path: str, output_path: str, coin_name: str) -> list[dict[str,Any]]:
    imgs = load_images_from_folder(input_path)
    if not imgs:
        logger.error("没有找到任何图片")
        sys.exit(1)
    os.makedirs(output_path, exist_ok=True)
    all_results = []
    for i, item in enumerate(imgs, start = 1):
        h, w = item['image'].shape[:2]
        if tuple(config_setting.calib_image_size) != (0, 0) and (w, h) != tuple(config_setting.calib_image_size):
            logger.warning(f"第{i}张图尺寸 {w}x{h} 与标定图 {tuple(config_setting.calib_image_size)} 不一致，"
                           f"ppm 不适用，直径会算错")
        logger.info(f"开始处理第{i}张图片...") 
        verify_ppm(i, item['image'])
        result_img, detections = process_single_frame(item['image'], coin_name)
        out_name = f"result_{i:03d}_{item['stem']}.jpg"
        if not image_io.write_image(os.path.join(output_path, out_name), result_img):
            logger.error(f"结果图写入失败: {out_name}") 
        all_results.append({
            'index': i,
            'image_name': item['name'],
            'detections': detections
        }) 
        logger.info(f"处理完成第{i}张图片")
    return all_results