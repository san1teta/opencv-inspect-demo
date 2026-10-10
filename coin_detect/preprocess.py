import cv2
import numpy as np
def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gaussian_blur(gray_img: np.ndarray, kernel_size: int=5) -> np.ndarray:
    return cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
def canny(blur_img: np.ndarray, low_threshold: int=50, high_threshold: int=150) -> np.ndarray:
    return cv2.Canny(blur_img, low_threshold, high_threshold)


def close_edges(edges: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """形态学闭运算：把断掉的边缘接上，只补断口，不整体加粗。

    为什么必须 CLOSE 而不是 DILATE —— 实拍数据（2026-10-10，5角 a 那枚三张连拍）：
        原图      窗内 0 个（币缘断成开口弧，RETR_EXTERNAL 算出的面积≈0）
        闭运算5   20.30 / 20.34 / 20.30 mm   ← 与同组完好币 20.35 一致
        膨胀5     20.89 / 21.03 mm           ← 系统性外扩 +0.6mm，是测量偏差不是找回
    膨胀把边缘带整体加粗，轮廓贴着粗带外沿走，直径被撑大 0.5~0.9mm —— 一项就吃掉
    允差 ±0.25mm 的两倍多。闭运算是「先膨胀再腐蚀」，腐蚀会把加粗的部分收回去，
    只有在断口处收不回去（那正是我们要接上的桥），所以边界基本原位。

    即便如此，接出来的桥仍然是一根「假边」，所以找回的轮廓只用于**定位与兜底测量**，
    pipeline 会在结果里标 edge_closed=True，让报表能看出这张是靠闭运算救回来的。
    """
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
