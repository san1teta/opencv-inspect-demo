"""统一的图片读写。

cv2.imread / cv2.imwrite 用 ANSI 打开文件，非 ASCII 路径（中文名、哈希名）会失败：
imread 返回 None、imwrite 返回 False，**都不抛异常**。手机传过来的照片十有八九是这种名字，
所以主管线必须走这里，否则就是「报表少几行但退出码 0」的静默丢图。
"""
import os

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """读图，支持非 ASCII 路径。读不到抛 FileNotFoundError，绝不返回 None 让调用方漏判。"""
    if path.isascii():
        img = cv2.imread(path)
        if img is not None:
            return img
    # 非 ASCII 路径直接走 imdecode：先试 imread 必然失败，还会往 stderr 喷一条
    # 「can't open/read file」的 WARN —— 手机传来的中文文件名一堆时，噪声能把真报错埋掉。
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> bool:
    """写图，支持非 ASCII 路径。返回是否成功 —— 调用方必须检查返回值。"""
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:      # 目录不存在 / 磁盘满 / 权限不足，都要变成 False 而不是抛出来
        return False
    return True
