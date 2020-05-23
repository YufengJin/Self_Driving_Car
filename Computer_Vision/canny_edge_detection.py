#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally

'''
kernel with large size can rough the image, reduce the noises for better canny filter 
'''
kernel_size = 11
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 0
high_threshold = 100
#https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)



# Display the image
plt.subplot(121)
plt.imshow(blur_gray, cmap = 'gray')
plt.title('Gaussian filter')

plt.subplot(122)
plt.imshow(edges, cmap='Greys_r')
plt.title('Canny Edge')

plt.show()