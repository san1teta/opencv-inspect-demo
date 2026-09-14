import cv2
import numpy as np
from typing import Any
color_map = {
    '合格': (0, 255, 0),
    '轻度': (0, 255, 255),
    '中度': (0, 165, 255),
    '重度': (0, 0, 255)
}

def draw_result(img: np.ndarray, all_features: list[dict[str, Any]]) -> np.ndarray:
    output_img = img.copy()
    for i in all_features:
        cnt = i['contour']
        x, y, w, h = cv2.boundingRect(cnt)
        if i['result']['circularity'] == '不合格':
            color = (0, 0, 255)
            label = '圆度不合格'
        else:
            color = color_map[i['result']['severity']]
            if i['result']['severity'] == '合格':
                label = '合格'
            else:
                label = f"{i['result']['direction']}-{i['result']['severity']}"

        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output_img