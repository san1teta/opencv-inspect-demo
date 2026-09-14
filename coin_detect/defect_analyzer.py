import config_setting
from typing import Any

def judge_quality(diameter_mm: float, circularity: float, coin_name: str) -> dict[str, Any]:
    result = {
        'circularity': '合格',
        'direction':'未检测',
        'severity':'未检测',
        'deviation':None
    }

    if circularity < config_setting.min_circularity:
        result['circularity'] = '不合格'
        return result

    spec = config_setting.coin_specs[coin_name]
    standard = spec['diameter']
    tol = spec['tol']

    deviation = round(diameter_mm - standard, 2)
    result['deviation'] = deviation
    excess = round(abs(deviation) - tol, 2)#滤除浮点垃圾
    if excess <= 0:
        result['severity'] = '合格'
        return result
    if excess > 0:
        result['direction'] = '偏大' if deviation>0 else '偏小'
        if excess <= config_setting.grade_light:
            result['severity'] = '轻度'
        elif excess <= config_setting.grade_mid:
            result['severity'] = '中度'
        else:
            result['severity'] = '重度'
    return result
