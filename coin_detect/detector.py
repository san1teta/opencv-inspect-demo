import cv2
import numpy as np

def find_contours(edge_img,min_area=100):
    contours = cv2.findContours(edge_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def calculate_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    circularity = (4*np.pi*area/perimeter**2) if perimeter > 0 else 0

    x,y,w,h = cv2.boundingRect(contour)
    (cx,cy),radius = cv2.minEnclosingCircle(contour)

    return{'area':area,
           'perimeter':perimeter,
           'circularity':circularity,
           'diameter_pixels':int(2*radius),
           'center':(int(cx),int(cy))
           }