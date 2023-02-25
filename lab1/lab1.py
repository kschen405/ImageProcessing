import cv2
import numpy as np
from histogram_equalization import histogram_equalization
from guassian_blur import guassian_blur
from median_filter import median_filter
from adjust_hsv import adjust_HSV


# -------------------------- test1.jpg ----------------------------------
img = cv2.imread('test1.jpg')
h, w, c = img.shape

# Histogram_Equalization
hq_img = histogram_equalization(img)

# Median filter
median_img = median_filter(hq_img, k_size=3)

# Comparison
compare_display = np.concatenate((img, median_img), axis=1)
cv2.imwrite("test1_median_3.jpg", compare_display)
cv2.imshow('test1_hq', compare_display)
cv2.waitKey(0)

# -------------------------- test2.jpg ----------------------------------

img = cv2.imread('test2.jpg')
img = cv2.resize(img, (512, 300))  
h, w, c = img.shape
print(img.shape)
# Histogram_Equalization
hq_img = histogram_equalization(img)

# HSI adjustment
adjust_img = adjust_HSV(hq_img, h_ratio=1, s_ratio=1, v_ratio=1.5)

# Comparison
compare_display = np.concatenate((img, adjust_img), axis=1)
cv2.imwrite("test2_hq_HSI_adjust.jpg", compare_display)
cv2.imshow('test1_hq', compare_display)
cv2.waitKey(0)

# -------------------------- test3.jpg ----------------------------------

img = cv2.imread('test3.jpg')
img = cv2.resize(img, (480, 480))  
h, w, c = img.shape
print(img.shape)

# Guassian blur
blur_img = guassian_blur(img, k_size=19)
compare_display = np.concatenate((img, blur_img), axis=1)
cv2.imwrite("test3_guassian_19.jpg", compare_display)
cv2.imshow('test1_hq', compare_display)
cv2.waitKey(0)


# -------------------------- test6.jpg ----------------------------------

img = cv2.imread('test6.jpg')
h, w, c = img.shape
img = cv2.resize(img, (480, 360))  
print(img.shape)

blur_img = guassian_blur(img, k_size=17)
medium_img = median_filter(blur_img, k_size=17)

# Histogram_Equalization
hq_img = histogram_equalization(medium_img)
# HSI
adjust = adjust_HSV(hq_img, 1, 1.5, 5.7)
# Comparison
compare_display = np.concatenate((img, adjust), axis=1)
cv2.imwrite("test6_hsv.jpg", compare_display)
cv2.imshow('test1_hq', compare_display)
cv2.waitKey(0)


