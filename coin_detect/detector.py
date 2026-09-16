import cv2
import numpy as np
from typing import Any

def find_contours(edge_img: np.ndarray, min_area: int = 100) -> list[np.ndarray]:
    contours = cv2.findContours(edge_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def calculate_features(contour:np.ndarray) -> dict[str, Any]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    circularity = (4*np.pi*area/perimeter**2) if perimeter > 0 else 0

    (cx,cy),radius = cv2.minEnclosingCircle(contour)

    return{'area':area,
           'perimeter':perimeter,
           'circularity':circularity,
           'diameter_pixels':int(2*radius),
           'center':(int(cx),int(cy))
           }