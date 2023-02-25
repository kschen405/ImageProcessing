import cv2
import numpy as np

def adjust_HSV(img, h_ratio=1, s_ratio=1, v_ratio=1):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print(hsv_img.shape)
    print(hsv_img[0,0,:])
    hsv_img[:,:,0] = hsv_img[:,:,0] * h_ratio 
    hsv_img[:,:,1] = np.where(hsv_img[:,:,1] * s_ratio <=255, hsv_img[:,:,1] * s_ratio, 255)
    hsv_img[:,:,2] = np.where(hsv_img[:,:,2] * v_ratio <=255, hsv_img[:,:,2] * v_ratio, 255)
    print(hsv_img[0,0,:])
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return hsv_img