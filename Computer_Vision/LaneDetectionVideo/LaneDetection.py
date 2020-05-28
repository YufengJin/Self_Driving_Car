import matplotlib.pyplot as plt
import cv2
import os
import numpy as np 
from LaneDetectionFunction import *
from moviepy.editor import VideoFileClip


def test_images_output(src):
    #laod image data
    image = cv2.imread(src)
    outcome = process_video(image)

    plt.imshow(combo)
    plt.title('Outcome')
    plt.show()


def process_video(image):
	imshape = image.shape
	gray = grayscale(image)

	kernel_size = 5
	low_threshold = 50
	high_threshold = 150
	blur_gray = gaussian_blur(gray,kernel_size)
	masked_edges = canny(blur_gray, low_threshold, high_threshold)

	vertices = np.array([[(0, imshape[0]), (465, 310), (475, 310), (imshape[1], imshape[0])]], dtype=np.int32)
	regioned_masked_edges = region_of_interest(masked_edges,vertices)

	rho = 2
	theta = np.pi / 180
	threshold = 45
	min_line_length = 40
	max_line_gap = 100

	line_image = np.copy(image)*0

	lines = cv2.HoughLinesP(regioned_masked_edges, rho, theta, threshold, np.array([]),
	                        min_line_length, max_line_gap)

	for line in lines:
	    for x1, y1, x2, y2 in line:
	        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

	color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
	combo = cv2.addWeighted(image, 0.8, line_image, 1, 0)

	return combo



if __name__ == '__main__':
	test_images = os.listdir("test_images/")

	'''
	plot all interior images during process
	'''

	#for image in test_images:
	#    path = os.path.join("test_images",image)
	#    test_images_output(path)
 
 	#process video test
	white_output = 'result_straight.mp4'
	clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
	white_clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)

	    






