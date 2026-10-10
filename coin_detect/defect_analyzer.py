import config_setting
from typing import Any

GRADES = ('合格', '轻度', '中度', '重度')


def _band(value: float, t1: float, t2: float, t3: float) -> int:
    """按三个门限把某个量映射成 0~3 档（0 = 在限内）。"""
    if value <= t1:
        return 0
    if value <= t2:
        return 1
    if value <= t3:
        return 2
    return 3


def judge_quality(diameter_mm: float, deficit_mm: float, circularity: float,
                  coin_name: str) -> dict[str, Any]:
    """两路独立判据，取更差的一档作总评：尺寸超差（直径）+ 边缘缺损（径向最大凹陷）。

    为什么必须两路 —— 实测数据（合成缺口，见 tools/selftest_measure.py 第 2 节，
    2026-10-10 在当前标定 ppm=8.6510 下重测）：
    啃掉 1mm 深的缺口，等效直径只变小 0.043mm，只有允差 0.25mm 的 1/6，尺寸路必然放过；
    同一枚的径向凹陷如实报出 0.992mm。反过来，整体磨小时径向凹陷恒为 ~0.11mm（噪声底），
    不会误报成刀口。两路各管各的，所以取 max 而不是相加。
    """
    result: dict[str, Any] = {
        'circularity': '合格',
        'direction':'未检测',
        'severity':'未检测',
        'deviation':None,
        'deficit_mm': None,
        'basis': '未检测'
    }

    spec = config_setting.coin_specs[coin_name]
    deviation = round(diameter_mm - spec['diameter'], 2)
    excess = round(abs(deviation) - spec['tol'], 2)#滤除浮点垃圾
    result['deviation'] = deviation
    result['deficit_mm'] = round(deficit_mm, 3)

    # 形状不合格是第三路判据，但**不再早退**。
    # 原来这里是 `if circularity < min: return result`，于是形状一不合格就把
    # deviation / deficit / severity 全部清空 —— 实拍里同一枚完好 1角 的三张连拍，
    # 因为轮廓上一根一像素毛刺把外接圆撑大 1.7mm，圆度从 0.977 掉到 0.824，
    # 那张好端端的数据在报表里就成了「未检测」，量出来的直径和缺损一起丢掉。
    # 圆度是「形状合不合格」的结论，不是「这张图能不能测」的开关；
    # 数值必须留下，否则 min_circularity 的分布永远攒不起来，而且误杀无法被发现。
    shape_fail = circularity < config_setting.min_circularity
    result['circularity'] = '不合格' if shape_fail else '合格'

    size_grade = _band(excess, 0, config_setting.grade_light, config_setting.grade_mid)
    edge_grade = _band(deficit_mm, config_setting.defect_light,
                       config_setting.defect_mid, config_setting.defect_heavy)
    shape_grade = 1 if shape_fail else 0     # 形状不合格至少算轻度
    grade = max(size_grade, edge_grade, shape_grade)
    result['severity'] = GRADES[grade]
    if grade == 0:
        result['basis'] = '无'
        result['direction'] = '未检测'
        return result

    # 三路可能同时命中同一档，那就都写进依据，别只报一个把人误导
    basis = []
    if size_grade == grade:
        basis.append('尺寸' + ('偏大' if deviation > 0 else '偏小'))
    if edge_grade == grade:
        basis.append(f'刀口{deficit_mm:.2f}mm')
    if shape_grade == grade:
        basis.append(f'形状{circularity:.3f}')
    result['basis'] = '+'.join(basis)
    if size_grade == grade:
        result['direction'] = '偏大' if deviation > 0 else '偏小'
    elif edge_grade == grade:
        result['direction'] = '缺损'
    else:
        result['direction'] = '形状'
    return result

