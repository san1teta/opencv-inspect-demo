import cv2
import numpy as np
from typing import Any
import config_setting

color_map = {
    '合格': (0, 255, 0),
    '轻度': (0, 255, 255),
    '中度': (0, 165, 255),
    '重度': (0, 0, 255)
}

# cv2.putText 只认 ASCII。喂中文它不报错，只画出「mmmnn」这种垃圾字形（实测就是这样，
# 整个 demo 的可视化输出等于没有）。所以图上用 ASCII 代号，中文分级仍然留在 CSV / Excel。
CODE = {
    '合格': 'OK',
    '轻度': 'L1',
    '中度': 'L2',
    '重度': 'L3',
}
DIRECTION = {'偏大': '+HI', '偏小': '-LO'}
SHAPE_NG = 'SHAPE-NG'


def draw_result(img: np.ndarray, all_features: list[dict[str, Any]]) -> np.ndarray:
    output_img = img.copy()
    w = output_img.shape[1]
    # 字号必须跟着画面走：写死 0.5 在 3072px 的原图上小到一个像素团，等于没标
    scale = max(0.5, w / 1200)
    thick = max(1, int(round(scale * 2)))
    for i in all_features:
        cnt = i['contour']
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        res = i['result']
        if res['circularity'] == '不合格':
            color = (0, 0, 255)
            label = SHAPE_NG
        else:
            color = color_map[res['severity']]
            label = f"{i['diameter_mm']:.2f}mm {CODE[res['severity']]}"
            label += DIRECTION.get(res['direction'], '')
            # 刀口单独标点位置：总评可能由尺寸档驱动，但缺损点仍要一眼看得到
            if res['deficit_mm'] is not None and res['deficit_mm'] > config_setting.defect_light:
                label += f" N{res['deficit_mm']:.2f}"
                dx, dy = i['deficit_point']
                cv2.circle(output_img, (int(dx), int(dy)), max(4, int(6 * scale)),
                           (0, 0, 255), thick)

        cv2.rectangle(output_img, (x, y), (x+w_rect, y+h_rect), color, thick)
        cv2.putText(output_img, label, (x, max(int(2.5 * scale * 12), y - 10 * thick)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return output_img
