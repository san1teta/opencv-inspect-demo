import cv2

def draw_result(img, all_features):
    output = img.copy()
    for i in all_features:
        cnt = i['contour']
        x, y, w, h = cv2.boundingRect(cnt)
        color = (0, 0,255) if i['defects'] else (0, 255, 0)
        label = i['coin_type']
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output