import cv2
import numpy as np

def guassian_blur(img, k_size=3):
    h, w, c = img.shape
    blur_img = img.copy()
    k = int((k_size - 1) / 2)
    for i in range(0, h):
        for j in range(0, w):
            sum = np.zeros(3)
            if i-k >= 0 and i+k < h:
                for u in range(-k, k+1):
                    if j-k >= 0 and j+k < w:
                        for v in range(-k, k+1):
                            sum += img[i+u, j+v]
            blur_img[i, j] = sum/(k_size*k_size)
    return blur_img