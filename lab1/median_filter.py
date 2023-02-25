import cv2
import numpy as np

def median_filter(img, k_size=3):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, c = img.shape
    median_img = img.copy()    
    for dim in range(img.ndim):
        k = int((k_size - 1) / 2)
        for i in range(0, h):
            for j in range(0, w):
                receptive_lst = []
                if i-k >= 0 and i+k < h:
                    for u in range(-k, k+1):
                        if j-k >= 0 and j+k < w:
                            for v in range(-k, k+1):
                                receptive_lst.append(img[i+u, j+v, dim]) 
                receptive_lst.sort()
                m = int(len(receptive_lst)/2)
                if receptive_lst != []:
                    median_img[i, j, dim] = receptive_lst[m]
    return median_img