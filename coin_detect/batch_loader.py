import os
import glob
import logging
import config_setting
import image_io
from calib_guard import stale_paths
from typing import Any

logger = logging.getLogger(__name__)

# 工具自己写回的标注图不能当待检样本。check_placement 默认把 place_*.jpg 落在输入目录里，
# 于是 glob('*.jpg') 会把它一起收进样本集 —— 一张画了红十字、绿圆、放大过的标注图混进
# 直径均值，重复性和线性度就全废了，而且不会报错，只会安静地给出错的数字。
# 下划线前缀是临时诊断图的约定，一并排除。
SKIP_PREFIXES = ('_', 'place_', 'result_')


def is_sample(name: str) -> bool:
    """这张图是不是待检样本（工具产物与临时诊断图不算）。"""
    return not name.startswith(SKIP_PREFIXES)

def load_images_from_folder(folder_path: str) -> list[dict[str, Any]]:
    if not os.path.isdir(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")   
        return []

    img_specs = ['*.jpg', '*.png', '*.bmp']
    
    paths = []
    for spec in img_specs:
        paths.extend(p for p in glob.glob(os.path.join(folder_path, spec))
                     if is_sample(os.path.basename(p)))
    paths.sort()
    # 比标定旧的照片会被当前 ppm 静默量出一个「看起来正常」的直径，混进均值和标准差。
    # 验收工具会 NG，但批量模式不替人决定这张图要不要 —— 所以这里**只警告、不丢弃**，
    # 同时把文件名摊出来，让人有机会在跑完之前喊停。判据见 calib_guard。
    stale = stale_paths(paths, getattr(config_setting, 'calib_stamp', ''))
    if stale:
        logger.warning(f"{len(stale)}/{len(paths)} 张比标定（{config_setting.calib_stamp}）旧，"
                       f"用当前 ppm 量它们没有意义，结果别写进报表："
                       + '、'.join(os.path.basename(p) for p in stale[:8])
                       + ('……' if len(stale) > 8 else ''))
    items = []
    for p in paths:
        try:
            img = image_io.read_image(p)
        except FileNotFoundError:
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