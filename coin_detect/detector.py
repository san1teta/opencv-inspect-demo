import cv2
import numpy as np
from typing import Any

def find_contours(edge_img: np.ndarray, min_area: int = 100) -> list[np.ndarray]:
    contours = cv2.findContours(edge_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def calculate_features(contour:np.ndarray) -> dict[str, Any]:
    """轮廓几何特征。两处刻意避开「看起来最直观」的算法，理由写在下面注释里。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)

    (cx,cy),radius = cv2.minEnclosingCircle(contour)

    # 1) 直径取面积等效值，不取外接圆直径。
    #    Canny 环的外多边形比真实边缘恒定外扩约 1.2px（ppm=9.97 下 = 0.12mm，
    #    19/20.5/25mm 三个尺寸实测都是 +1.21px，是系统偏差不是噪声），一项就吃光
    #    0.1mm 预算；面积等效直径实测误差 +0.014mm。
    #    更要紧的是外接圆对缺料不敏感：刀口啃掉一块，外接圆仍是原来那个圆，
    #    而本 demo 要检的正是边缘缺损。
    diameter_pixels = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

    # 2) 圆度把周长换成外接圆周长：4πA/(2πr)² = A/(πr²)，与经典定义同源。
    #    数字轮廓的周长被阶梯效应撑高约 5%，4πA/P² 连数学上完美的圆都只有 0.899，
    #    门槛 0.9 按构造就跨不过去（= 任何硬币一律判不合格）；换成 A/(πr²) 后
    #    完美圆 ≈ 0.99，只有真缺料才会掉下来。
    envelope = np.pi * radius ** 2
    circularity = (area / envelope) if envelope > 0 else 0

    return{'area':area,
           'perimeter':perimeter,
           'circularity':float(circularity),
           'diameter_pixels':diameter_pixels,
           'center':(int(cx),int(cy))
           }