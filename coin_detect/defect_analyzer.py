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

    为什么必须两路 —— 实测数据（合成缺口，见 tools/selftest_measure.py 第 2 节）：
    啃掉 1mm 深的缺口，等效直径只变小 0.035mm，只有允差 0.25mm 的 1/7，尺寸路必然放过；
    同一枚的径向凹陷如实报出 1.001mm。反过来，整体磨小时径向凹陷恒为 ~0.10mm（噪声底），
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

    if circularity < config_setting.min_circularity:
        result['circularity'] = '不合格'
        return result

    spec = config_setting.coin_specs[coin_name]
    deviation = round(diameter_mm - spec['diameter'], 2)
    excess = round(abs(deviation) - spec['tol'], 2)#滤除浮点垃圾
    result['deviation'] = deviation
    result['deficit_mm'] = round(deficit_mm, 3)

    size_grade = _band(excess, 0, config_setting.grade_light, config_setting.grade_mid)
    edge_grade = _band(deficit_mm, config_setting.defect_light,
                       config_setting.defect_mid, config_setting.defect_heavy)
    grade = max(size_grade, edge_grade)
    result['severity'] = GRADES[grade]
    if grade == 0:
        result['basis'] = '无'
        return result

    # 两路可能同时命中同一档，那就都写进依据，别只报一个把人误导
    basis = []
    if size_grade == grade:
        basis.append('尺寸' + ('偏大' if deviation > 0 else '偏小'))
    if edge_grade == grade:
        basis.append(f'刀口{deficit_mm:.2f}mm')
    result['basis'] = '+'.join(basis)
    result['direction'] = ('偏大' if deviation > 0 else '偏小') if size_grade == grade else '缺损'
    return result

