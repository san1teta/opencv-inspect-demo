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

    # 径向量的参考圆心取轮廓质心，不取外接圆圆心。
    # 外接圆是「最大值」估计量，只由 2~3 个极端点决定：真币轮廓上任何一处毛刺都会把
    # 圆心整个拖走。实测一枚完好 1元，外接圆圆心比质心偏 5.1px，绕着偏心圆心量出的
    # deficit 平白多出 0.54mm（0.690 vs 0.148）——直接越过 defect_light=0.4，
    # 把每一枚良品判成有缺口。合成完美圆测不出这个问题，因为完美圆的外接圆圆心恰好
    # 就是真圆心；只有真币轮廓才会暴露它。
    # 代价：材料被啃掉后质心会朝缺口反方向偏，缺口越深低估越多（3mm 真值报 2.59，-14%），
    # 但 0.5/1.0/2.0/3.0mm 实测 0.49/0.97/1.83/2.59 仍分别落在轻/中/重/重四档内，
    # 分档不受影响。换来的是良品侧 0.25mm 余量 —— 宁可少报深度，不可凭空造缺陷。
    mom = cv2.moments(contour)
    if mom['m00'] > 0:
        cx, cy = mom['m10'] / mom['m00'], mom['m01'] / mom['m00']

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

    # 3) 径向最大凹陷：刀口 / 缺口的直接量。
    #    直径法对缺口几乎无感 —— 实测 1mm 深的缺口只让等效直径变小 0.035mm（面积是全局量，
    #    局部少一小块摊到整个圆上就没了），而缺口是局部事件，就必须用局部量去抓。
    #    参考半径取「中位半径」而不是外接圆半径：整体磨小是中位半径一起缩，凹陷仍是 0，
    #    于是「尺寸超差」和「边缘缺损」两件事被干净分开，各走各的判据。
    pts = contour.reshape(-1, 2).astype(np.float32)
    dist = np.linalg.norm(pts - np.array([cx, cy], np.float32), axis=1)
    base = float(np.median(dist))
    worst = int(np.argmin(dist))
    deficit_pixels = max(0.0, base - float(dist[worst]))

    return{'area':area,
           'perimeter':perimeter,
           'circularity':float(circularity),
           'diameter_pixels':diameter_pixels,
           'deficit_pixels':deficit_pixels,
           'deficit_point':(float(pts[worst][0]), float(pts[worst][1])),
           'center':(int(cx),int(cy))
           }


def diameter_window(nominal_mm: float, ppm: float, gate_mm: float) -> tuple[float, float]:
    """该面额硬币的等效直径像素窗口 [lo, hi]，落在窗外的轮廓是背景不是币。

    返回像素而不是毫米，是为了让调用方拿 calculate_features 的 'diameter_pixels'
    直接比 —— 这个模块此前一直只做像素域几何，不引进 ppm。
    窗口开在 ±2mm（config_setting.diameter_gate_mm），比允差 0.25mm 宽 8 倍，这是故意的：
    它的职责只有「这是不是枚币」，「这枚币合不合格」仍归 defect_analyzer 判。两者不能共用
    一个阈值，否则尺寸超差的币会被当成背景悄悄丢掉，而不是如实判成重度不合格。
    """
    return ((nominal_mm - gate_mm) * ppm, (nominal_mm + gate_mm) * ppm)
