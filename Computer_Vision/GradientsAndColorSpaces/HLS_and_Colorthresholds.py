import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test6.jpg')
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

# Gray binary
plt.imshow(binary,cmap='gray')
plt.title('Gray Binary')

# seperate into RGB three channels
R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]

'''
show images in three color channel
'''
plt.subplot(131)
plt.imshow(R,cmap='gray')
plt.title('R')
plt.subplot(132)
plt.imshow(G,cmap='gray')
plt.title('G')
plt.subplot(133)
plt.imshow(B,cmap='gray')
plt.title('B')


thresh = (200, 255)
binary_R = np.zeros_like(R)
binary_R[(R > thresh[0]) & (R <= thresh[1])] = 1

#show R channel and R binary 
plt.subplot(121)
plt.imshow(R,cmap='gray')
plt.title('R Channel')
plt.subplot(122)
plt.imshow(binary,cmap='gray')
plt.title('R Binary')


#HLS convertion
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

'''
show images in three HLS Channels
'''
plt.subplot(131)
plt.imshow(H,cmap='gray')
plt.title('H')
plt.subplot(132)
plt.imshow(L,cmap='gray')
plt.title('L')
plt.subplot(133)
plt.imshow(S,cmap='gray')
plt.title('S')

'''
The S channel picks up the lines well, so let's try applying a threshold there:
'''
thresh = (90, 255)
binary_S = np.zeros_like(S)
binary_S[(S > thresh[0]) & (S <= thresh[1])] = 1

# show S Channel and S Binary
plt.subplot(121)
plt.imshow(S,cmap='gray')
plt.title('S Channel')
plt.subplot(122)
plt.imshow(binary_S,cmap='gray')
plt.title('S Binary')
plt.show()


'''
You can also see that in the H channel, the lane lines appear dark, so we could try a low threshold there and obtain the following result:
'''

thresh = (15, 100)
binary = np.zeros_like(H)
binary[(H > thresh[0]) & (H <= thresh[1])] = 1