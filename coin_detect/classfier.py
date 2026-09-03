from config_setting import coin_standards

def classify_coin(diameter_mm):
    for name, std_dia in coin_standards.items():
        if abs(diameter_mm - std_dia) < 0.2:
            return name
    return '未知'