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

def _nominal(coin_name: str) -> float | None:
    """这枚面额的标称直径；认不出面额返回 None（检出闸门随之降级，见 coin_candidates）。"""
    spec = config_setting.coin_specs.get(coin_name)
    return spec['diameter'] if spec else None


def _extract(edge_img: np.ndarray, coin_name: str) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], list[dict[str, Any]]]:
    """从一张边缘图里取出「像这枚面额」的轮廓，同时把全部特征带回来供诊断用。"""
    contours = detector.find_contours(edge_img, config_setting.min_area)
    feats = [detector.calculate_features(c) for c in contours]
    kept = detector.coin_candidates(feats, _nominal(coin_name), config_setting.ppm,
                                    config_setting.diameter_gate_mm,
                                    config_setting.min_circularity)
    # 用 id 集合回筛：feats 与 contours 同序构造，本次调用内对象唯一，不会串。
    ids = {id(f) for f in kept}
    return [(c, f) for c, f in zip(contours, feats) if id(f) in ids], feats


def find_coins(img: np.ndarray, coin_name: str) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], bool]:
    """找币的唯一入口：预处理 → Canny → 轮廓 → 尺寸窗，币缘断开时自动闭运算兜底。

    返回 ((轮廓, 特征) 列表, 是否走了闭运算兜底)。流水线、report_metrics、
    check_placement 三处都从这里进 —— 判据只有一份，不会再出现「一条路修好了、
    另一条路还在用旧规则」那种事（2026-10-09 的尺寸闸门就是这么漏的）。
    """
    gray = preprocess.gray(img)
    blur = preprocess.gaussian_blur(gray, config_setting.gaussian_kernel)
    edges = preprocess.canny(blur, config_setting.canny_low, config_setting.canny_high)
    pairs, feats = _extract(edges, coin_name)
    if pairs:
        return _cross_check(pairs, blur), False
    if not feats:
        return [], False                      # 连轮廓都没有，闭运算也变不出东西来
    # 币缘被 Canny 的滞后高门限切成开口弧 —— RETR_EXTERNAL 对开口弧算出的面积≈0，
    # 等效直径只剩几毫米，于是全部落在尺寸窗外，表现为「画面里没有币」。
    # 实拍复核证明币一直在：那 6 张「未检出」的币缘梯度中位 236~302，是 Canny 高门限
    # 150 的 1.6~2 倍；同一枚 1角 同一秒的三张里 b03 闭合、b01/b02 断开。
    # 断口取决于圆周上的受光分布，差一个像素就换结果 —— 靠重拍解决不了，必须兜底。
    # 用闭运算而不是膨胀：膨胀会把直径系统性撑大 0.5~0.9mm（见 preprocess.close_edges）。
    closed, _ = _extract(preprocess.close_edges(edges), coin_name)
    if closed:
        logger.info(f"{coin_name}：原图币缘断成开口弧检不到，闭运算找回 {len(closed)} 个目标")
        return _cross_check(closed, blur), True
    nominal = _nominal(coin_name)
    if nominal:
        # 全军覆没最可能是标定/距离出了问题，而不是「画面里没有币」，所以把最大候选的
        # 实际直径写进日志，别只留一句「未检出」让人去猜是哪一类错。
        big = max(feats, key=lambda f: f['area'])
        logger.warning(f"{coin_name}：{len(feats)} 个轮廓（含闭运算兜底）全部落在标称 "
                       f"{nominal}±{config_setting.diameter_gate_mm}mm 之外"
                       f"（最大者等效直径 {big['diameter_pixels'] / config_setting.ppm:.2f}mm、"
                       f"圆度 {big['circularity']:.3f}）—— 先查相机距离与 ppm，再确认币有没有入画")
    return [], False


def _cross_check(pairs: list[tuple[np.ndarray, dict[str, Any]]],
                 blur: np.ndarray) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """并跑第二路读数（亚像素径向）做**互检**，不改上报值，只产生告警。

    为什么是互检而不是替换：多边形读数在干净币上偏差只有 −0.005~−0.08mm，够好；
    但它会被高边圈的**内台阶**骗，静默读小 1mm（j 那枚 22.25 读成 21.188，圆度 0.966
    看起来完全健康，任何质量判据都发现不了）。径向法能看见那道内台阶之外的真实外缘，
    但它自己在干净币上又系统性偏 −0.10mm、还会被眩光晕过冲 —— 当上报值不合格，
    当**分歧探测器**刚好：两路分叉就说明有一路被几何骗了。
    """
    out = []
    for cnt, f in pairs:
        rm = detector.radial_measure(blur, f['center'], f['diameter_pixels'] / 2)
        if not rm:
            out.append((cnt, {**f, 'radial_mm': None, 'edge_suspect': False}))
            continue
        rad_mm = rm['radial_diameter_pixels'] / config_setting.ppm
        poly_mm = f['diameter_pixels'] / config_setting.ppm
        gap = abs(rad_mm - poly_mm)
        suspect = gap > config_setting.edge_check_mm
        out.append((cnt, {**f, **rm, 'radial_mm': rad_mm, 'edge_suspect': suspect}))
        if suspect:
            logger.warning(f"轮廓可疑：多边形 {poly_mm:.3f}mm vs 径向 {rad_mm:.3f}mm "
                           f"差 {gap:.3f}mm > {config_setting.edge_check_mm}mm —— "
                           f"上报值仍用多边形，但这个数别直接采信（轮廓可能被内台阶/眩光骗）")
    return out


def process_single_frame(frame: np.ndarray, coin_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """单帧定义"""
    pairs, edge_closed = find_coins(frame, coin_name)
    all_features = []
    for cnt, features in pairs:
        diameter_mm = calibration.pixels_to_mm(features['diameter_pixels'], config_setting.ppm)
        deficit_mm = calibration.pixels_to_mm(features['deficit_pixels'], config_setting.ppm)
        result = defect_analyzer.judge_quality(diameter_mm, deficit_mm,
                                               features['circularity'], coin_name)
        all_features.append({
            'contour': cnt,
            'diameter_mm': diameter_mm,
            # 齿顶对径（与卡尺同口径）单独留着：面积等效把齿槽/内台阶当成「少掉的面积」。
            # 尺寸合格判定早晚要换到这一列，先把数存下来才能对比。
            'caliper_mm': calibration.pixels_to_mm(features['caliper_diameter_pixels'],
                                                   config_setting.ppm),
            # 互检两列：径向读数是**第二路独立测量**，只对照不参与判定；
            # edge_suspect = 两路分叉超阈值，edge_closed = 靠闭运算补断口才定位到。
            # 没有这两列，j 那种「静默读小 1mm、圆度还挺健康」的帧会在报表里一路「合格」。
            'radial_mm': features.get('radial_mm'),
            'edge_suspect': features.get('edge_suspect', False),
            # 数值圆度单独留一份：judge_quality 把 result['circularity'] 覆成了
            # 「合格/不合格」，光看报表拿不到分布，而 min_circularity=0.90 这条临时
            # 门限正是靠真样本的分布来定的。
            'circularity': features['circularity'],
            'deficit_point': features['deficit_point'],
            # 轮廓是靠闭运算补断口找回来的 —— 必须让报表看得见，否则「兜底成功」
            # 和「正常检出」混在同一列里，直径略偏也无人知情。
            'edge_closed': edge_closed,
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