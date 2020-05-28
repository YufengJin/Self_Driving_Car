import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread("test.jpg")

print('This image is: ',type(image), 
         'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

'''
Next I define a color threshold in the variables red_threshold, green_threshold, and blue_threshold 
and populate rgb_threshold with these values. This vector contains the minimum values for red, green, 
and blue (R,G,B) that I will allow in my selection.
'''

red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]


# or function, get pixel in three channels within threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# the threshold is True or False ndarray with dimension of 540*960
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()