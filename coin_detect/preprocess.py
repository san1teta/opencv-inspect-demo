import cv2
import numpy as np
def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gaussian_blur(gray_img: np.ndarray, kernel_size: int=5) -> np.ndarray:
    return cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
def canny(blur_img: np.ndarray, low_threshold: int=50, high_threshold: int=150) -> np.ndarray:
    return cv2.Canny(blur_img, low_threshold, high_threshold)
