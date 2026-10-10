import cv2
import numpy as np
from typing import Any

def find_contours(edge_img: np.ndarray, min_area: int = 100) -> list[np.ndarray]:
    contours = cv2.findContours(edge_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

_SECTORS = 360

# 检出这一层的几何地板：圆度低于它的轮廓不是「一个填充区域」，是一根细环/碎边。
# 定在 0.35 是量出来的，不是拍的（2026-10-10 实拍 40 张）：
#     真币（闭运算兜底找回、轮廓已被桥污染）  圆度 0.386 ~ 0.434
#     横跨画面的细环垃圾                       圆度 0.004，r_ref 165mm，缺损 57~61mm
# 0.35 把两类分开，同时离两边都有余量：对真币留 0.036 的容差，对垃圾留 100 倍。
# 半枚币的 面积/外接圆面积 正好 0.5，所以这条地板杀不掉任何 conceivable 的硬币。
# 与质量门限 config_setting.min_circularity(=0.9) 分工不同，别合并成一个数。
REGION_FLOOR = 0.35


def _crest_radii(dist: np.ndarray, ang: np.ndarray, sectors: int = _SECTORS) -> np.ndarray:
    """把轮廓点按方位角分成 sectors 个扇区，每扇区取最外缘半径 = 该方向的齿顶半径。

    用「最外」而不是「平均」：齿槽是凹进去的，平均半径落在齿根与齿顶之间，
    谁都不是。卡尺夹的是齿顶，所以这里也必须取齿顶。
    空扇区用两侧最近的非空扇区线性插值补上（轮廓点通常 1000+，极少出现空洞）。
    """
    idx = np.floor((ang + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    crest = np.zeros(sectors, np.float64)
    np.maximum.at(crest, idx, dist)
    filled = np.flatnonzero(crest > 0)
    if filled.size == 0:
        return crest
    if filled.size < sectors:                    # 环形线性插值，避免空扇区把对径读成 0
        pos = np.arange(sectors)
        crest = np.interp(pos, np.concatenate([filled, filled + sectors]),
                          np.concatenate([crest[filled], crest[filled]])).astype(np.float64)
    return crest


def calculate_features(contour:np.ndarray) -> dict[str, Any]:
    """轮廓几何特征。两处刻意避开「看起来最直观」的算法，理由写在下面注释里。"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)

    (cx, cy), _enclosing = cv2.minEnclosingCircle(contour)

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

    pts = contour.reshape(-1, 2).astype(np.float32)
    dist = np.linalg.norm(pts - np.array([cx, cy], np.float32), axis=1)

    # 1) 直径取面积等效值，不取外接圆直径。
    #    Canny 环的外多边形比真实边缘恒定外扩约 1.2px（ppm=9.97 下 = 0.12mm，
    #    19/20.5/25mm 三个尺寸实测都是 +1.21px，是系统偏差不是噪声），一项就吃光
    #    0.1mm 预算；面积等效直径实测误差 +0.014mm。
    #    更要紧的是外接圆对缺料不敏感：刀口啃掉一块，外接圆仍是原来那个圆，
    #    而本 demo 要检的正是边缘缺损。
    diameter_pixels = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

    # 2) 圆度 = 面积 / 参考圆面积。参考半径改用「半径分布的 99.5 分位」，不用
    #    minEnclosingCircle 的半径 —— 外接圆是最大值估计量，只由最外侧 2~3 个点决定。
    #    实拍证据（同一枚完好 1角、相隔一秒的连拍）：面积只差 0.1%（20690 vs 20710 px），
    #    外接圆直径却从 18.98mm 跳到 20.68mm，圆度 0.977 → 0.824 → 好币被判成形状不合格。
    #    起因是轮廓上根一像素宽的毛刺桥接到背景边缘，把外接圆整个撑大 1.7mm。
    #    98 分位丢掉最外侧 2% 的点（轮廓点约 1000 个，即允许 20 个点离群）—— 一根毛刺桥
    #    在轮廓上留下的正是十几个点这个量级；99.5 分位只排除 5 个点，实测滤不掉。
    #    而真缺口是「一整段弧向内凹」，影响的是半径分布的**内侧**，不动上侧分位，
    #    所以缺口照样如实反映在面积上。
    r_ref_pixels = float(np.percentile(dist, 98)) if dist.size else 0.0
    envelope = np.pi * r_ref_pixels ** 2
    circularity = (area / envelope) if envelope > 0 else 0

    # 3) 齿顶对径（卡尺等效）。面积等效直径把齿槽当成「少掉的面积」，量出来是齿根侧：
    #    卡尺实测一枚 1元 = 22.25mm，而面积等效只有 22.04mm（-0.21mm，占满允差的 84%），
    #    且这个偏差逐枚不同（5角 -0.07~-0.15，1角 -0.17~-0.29），平移允差中心修不了。
    #    卡尺量的是两个平行 jaw 之间的最大距离，所以这里按方位角分 360 个扇区取每向的
    #    最外缘半径（齿顶），再取对向之和的高分位 —— 与卡尺同口径，且天然抗单点毛刺。
    crest = _crest_radii(dist, np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx), _SECTORS)
    half = _SECTORS // 2
    widths = crest[:half] + crest[half:]
    caliper_diameter_pixels = float(np.percentile(widths, 99)) if widths.size else 0.0

    # 4) 径向最大凹陷：刀口 / 缺口的直接量。
    #    直径法对缺口几乎无感 —— 实测 1mm 深的缺口只让等效直径变小 0.035mm（面积是全局量，
    #    局部少一小块摊到整个圆上就没了），而缺口是局部事件，就必须用局部量去抓。
    #    参考半径取「中位半径」而不是外接圆半径：整体磨小是中位半径一起缩，凹陷仍是 0，
    #    于是「尺寸超差」和「边缘缺损」两件事被干净分开，各走各的判据。
    base = float(np.median(dist))
    worst = int(np.argmin(dist))
    deficit_pixels = max(0.0, base - float(dist[worst]))

    return{'area':area,
           'perimeter':perimeter,
           'circularity':float(circularity),
           'diameter_pixels':diameter_pixels,
           'caliper_diameter_pixels':caliper_diameter_pixels,
           'r_ref_pixels':r_ref_pixels,
           'deficit_pixels':deficit_pixels,
           'deficit_point':(float(pts[worst][0]), float(pts[worst][1])),
           'center':(int(cx),int(cy))
           }


def coin_candidates(feats: list[dict[str, Any]], nominal_mm: float | None, ppm: float,
                    gate_mm: float, min_circularity: float) -> list[dict[str, Any]]:
    """从特征列表里挑「像是这枚面额的币」。**判据只写这一份**，流水线 / 报表 / 验收工具共用。

    为什么必须共用：2026-10-09 尺寸闸门只长在 check_placement 上、没长在流水线上，
    于是同一张图一个报 1 个目标、另一个报 51 个「币」；2026-10-10 补了闭运算兜底之后，
    如果只改流水线，report_metrics 的检出率就还是按旧判据算 —— 同一个坑换个位置再踩。

    有面额 → 用直径窗口（「这是不是枚币」）。
    没面额 → 没有标称可依，退回圆度筛。这是**降级**不是等价方案，调用方要明说。
    圆度低是「这枚币变形了」，属质量结论，绝不能在检出这一层被悄悄删掉。

    但尺寸窗只回答「多大」，不回答「还是个填充区域吗」。闭运算补断口时，偶尔会把碎边
    连成一根横跨画面的细环：它面积摊出来恰好落在窗内，圆度却只有 0.004 —— 实测这种
    轮廓被「取面积最大」选中后，缺损报出 59mm、圆度 0.004，1角 的重复性 σ 从 0.003
    被顶到 0.628。所以这里加一条**几何 sanity 地板** REGION_FLOOR。

    它和质量门限 min_circularity 是两件事，不能混：
        REGION_FLOOR = 0.35   回答「这还是不是一个填充区域」—— 半枚币的 面积/外接圆面积
                              正好是 0.5，所以这条地板杀不掉任何 conceivable 的硬币，
                              只杀得掉「根本不是区域」的细环。
        min_circularity = 0.9 回答「这枚币圆不圆」—— 质量结论，只能判不合格，不许删数据。
    """
    if nominal_mm:
        lo, hi = diameter_window(nominal_mm, ppm, gate_mm)
        return [f for f in feats
                if lo <= f['diameter_pixels'] <= hi and f['circularity'] >= REGION_FLOOR]
    return [f for f in feats if f['circularity'] >= min_circularity]


def diameter_window(nominal_mm: float, ppm: float, gate_mm: float) -> tuple[float, float]:
    """该面额硬币的等效直径像素窗口 [lo, hi]，落在窗外的轮廓是背景不是币。

    返回像素而不是毫米，是为了让调用方拿 calculate_features 的 'diameter_pixels'
    直接比 —— 这个模块此前一直只做像素域几何，不引进 ppm。
    窗口开在 ±2mm（config_setting.diameter_gate_mm），比允差 0.25mm 宽 8 倍，这是故意的：
    它的职责只有「这是不是枚币」，「这枚币合不合格」仍归 defect_analyzer 判。两者不能共用
    一个阈值，否则尺寸超差的币会被当成背景悄悄丢掉，而不是如实判成重度不合格。
    """
    return ((nominal_mm - gate_mm) * ppm, (nominal_mm + gate_mm) * ppm)
