import cv2
import numpy as np

def histogram_equalization(img): # images with single channel
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    intensity = hsv_img[:, :, 2]
    histogram = np.bincount(intensity.flatten(), minlength=256)
    num_pixel = np.sum(histogram)
    histogram = histogram / num_pixel
    chistogram = np.cumsum(histogram)

    transform_map = np.floor(255 * chistogram).astype(np.uint8)
    old_intensity = list(intensity.flatten())
    new_intensity = [transform_map[i] for i in old_intensity]
    hq_intensity = np.reshape(np.asarray(new_intensity), intensity.shape)
    hsv_img[:, :, 2] = hq_intensity
    hq_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return hq_img
