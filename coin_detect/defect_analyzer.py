def judge_quality(features, min_circularity, min_diameter):
    defects = []
    if features['circularity'] < min_circularity:
        defects.append("变形")
    if features['diameter_pixels'] < min_diameter:
        defects.append("磨损")
        
    return defects if defects else None

def analyze_defects(features):
    if features['diameter_pixels'] > 20:
        min_diameter = 10
        min_circularity = 0.5
        coin_type = "1元"
    elif features['diameter_pixels'] > 10:
        min_diameter = 5
        min_circularity = 0.3
        coin_type = "5角"
    else:
        min_diameter = 2
        min_circularity = 0.1
        coin_type = "1角"
    
    defects = judge_quality(features, min_circularity, min_diameter)
    return defects, coin_type
