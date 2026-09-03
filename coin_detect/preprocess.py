import cv2
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gaussian_blur(gray_img, kernel_size=5):
    return cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
def canny(blur_img, low_threshold=50, high_threshold=150):
    return cv2.Canny(blur_img, low_threshold, high_threshold)
