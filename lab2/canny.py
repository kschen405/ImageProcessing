import cv2
import numpy as np
import matplotlib.pyplot as plt 



def convolve(img, kernel):
    h, w = img.shape
    k_size = kernel.shape[0]
    pad_one_side = (k_size - 1)//2
    h_pad = h + 2*pad_one_side
    w_pad = w + 2*pad_one_side
    
    img_pad = np.zeros((h_pad, w_pad))
    # print(img_pad.shape)
    # print(img.shape)
    # print(img_pad[pad_one_side : -pad_one_side, pad_one_side : -pad_one_side].shape)
    img_pad[pad_one_side : -pad_one_side, pad_one_side : -pad_one_side] = img

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mat = img_pad[i : i + k_size, j : j + k_size]
            # print("mat = ", mat)
            img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return img


class Canny:
    def __init__(self, imgs, threshold1, threshold2, gaussian_sigma=1, gaussian_k_size=5, strong=255, weak=32):
        self.img_lst = imgs
        self.threshold1 =threshold1
        self.threshold2 = threshold2
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_k_size = gaussian_k_size
        self.strong = strong
        self.weak = weak

    def gaussian_kernel(self, k_size=5, sigma=1):
        tmp = (k_size - 1) / 2
        x, y = np.mgrid[-tmp:tmp+1, -tmp:tmp+1]
        kernel =  1 / (2.0 * np.pi * sigma*sigma) * np.exp(-((x*x + y*y) / (2.0*sigma*sigma)))
        return kernel

    def gaussian_blur(self, img, k_size=5, sigma=1):
        kernel = self.gaussian_kernel(k_size, sigma)
        img_blur = convolve(img, kernel)
        return img_blur

    def sobel_edge(self, img):
        sx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
        sy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
        gradient_x = convolve(img, sx)
        gradient_y = convolve(img, sy)
        magnitude = np.hypot(gradient_x, gradient_y)
        magnitude = magnitude / magnitude.max() *255
        magnitude = magnitude.astype(np.uint8)
        theta = np.arctan2(gradient_y, gradient_x)
        return magnitude, theta

    def non_max_suppression(self, magnitude, theta):
        h, w = magnitude.shape
        output_magnitude = np.zeros(magnitude.shape)
        angle = np.rad2deg(theta)
        angle[angle < 0] += 180
        PI = 180
        for row in range(1, h-1):
            for col in range(1, w-1):
                direction = angle[row, col]
                if (0 <= direction < PI/8) or (7*PI/8 <= direction <= PI):
                    neighbor1 = magnitude[row][col+1]
                    neighbor2 = magnitude[row][col-1]
                elif (PI/8 <= direction < 3*PI/8):
                    neighbor1 = magnitude[row+1][col-1]
                    neighbor2 = magnitude[row-1][col+1]
                elif (3*PI/8 <= direction < 5*PI/8):
                    neighbor1 = magnitude[row+1][col]               
                    neighbor2 = magnitude[row-1][col]
                elif (5*PI/8 <= direction < 7*PI/8):               
                    neighbor1 = magnitude[row-1][col-1]               
                    neighbor2 = magnitude[row+1][col+1]               

                if magnitude[row, col] > neighbor1 and magnitude[row, col] > neighbor2:
                    output_magnitude[row, col] = magnitude[row, col]
        return output_magnitude
    
    def hysteresis_thresholding(self, magnitude):
        h, w = magnitude.shape
        output = np.zeros(magnitude.shape)
        strong_row, strong_col = np.where(self.threshold1 < magnitude)
        weak_row, weak_col = np.where((self.threshold2 < magnitude) & (magnitude <= self.threshold1))
        output[strong_row, strong_col] = self.strong
        output[weak_row, weak_col] = self.weak

        top2bottom = output.copy()
        for row in range(1, h-1):
            for col in range(1, w-1):
                if top2bottom[row][col] == self.weak:
                    if top2bottom[row+1][col+1] ==self.strong or top2bottom[row+1][col] ==self.strong or top2bottom[row+1][col-1] ==self.strong or top2bottom[row][col+1] ==self.strong or top2bottom[row][col-1] ==self.strong or top2bottom[row-1][col+1] ==self.strong or top2bottom[row-1][col] ==self.strong or top2bottom[row-1][col-1] ==self.strong:
                        top2bottom[row, col] = self.strong
                    else:
                        top2bottom[row, col] = 0

        bottom2top = output.copy()
        for row in range(h-2, 0, -1):
            for col in range(w-2, 0, -1):
                if bottom2top[row][col] == self.weak:
                    if bottom2top[row+1][col+1] ==self.strong or bottom2top[row+1][col] ==self.strong or bottom2top[row+1][col-1] ==self.strong or bottom2top[row][col+1] ==self.strong or bottom2top[row][col-1] ==self.strong or bottom2top[row-1][col+1] ==self.strong or bottom2top[row-1][col] ==self.strong or bottom2top[row-1][col-1] ==self.strong:
                        bottom2top[row, col] = self.strong
                    else:
                        bottom2top[row, col] = 0

        right2left = output.copy()
        for row in range(1, h-1):
            for col in range(w-2, 0, -1):
                if right2left[row][col] == self.weak:
                    if right2left[row+1][col+1] ==self.strong or right2left[row+1][col] ==self.strong or right2left[row+1][col-1] ==self.strong or right2left[row][col+1] ==self.strong or right2left[row][col-1] ==self.strong or right2left[row-1][col+1] ==self.strong or right2left[row-1][col] ==self.strong or right2left[row-1][col-1] ==self.strong:
                        right2left[row, col] = self.strong
                    else:
                        right2left[row, col] = 0

        left2right = output.copy()
        for row in range(h-2, 0, -1):
            for col in range(1, w-1):
                if left2right[row][col] == self.weak:
                    if left2right[row+1][col+1] ==self.strong or left2right[row+1][col] ==self.strong or left2right[row+1][col-1] ==self.strong or left2right[row][col+1] ==self.strong or left2right[row][col-1] ==self.strong or left2right[row-1][col+1] ==self.strong or left2right[row-1][col] ==self.strong or left2right[row-1][col-1] ==self.strong:
                        left2right[row, col] = self.strong
                    else:
                        left2right[row, col] = 0
        sum_results = top2bottom + bottom2top + right2left + left2right
        sum_results[sum_results > 255] = 255

        return sum_results
        
    def detect_edge(self):
        output_lst = []
        for img in self.img_lst:
            img_blur = self.gaussian_blur(img, self.gaussian_k_size, self.gaussian_sigma)
            magnitude, theta = self.sobel_edge(img_blur)
            magnitude_nms = self.non_max_suppression(magnitude, theta)
            output = self.hysteresis_thresholding(magnitude_nms)
            output_lst.append(output)
        return output_lst

img = cv2.imread("data/2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = Canny(img, threshold1= 200, threshold2= 100, gaussian_sigma=1, gaussian_k_size=5 )
output = canny.gaussian_blur(img, 5, 1.5)
print(type(output[0][0]))
print(output.shape)
output, theta = canny.sobel_edge(output)

print(type(output[0][0]))
# magnitude_nms = canny.non_max_suppression(output, theta)
plt.imshow(output, cmap="gray")
# cv2.waitKey(0)
        