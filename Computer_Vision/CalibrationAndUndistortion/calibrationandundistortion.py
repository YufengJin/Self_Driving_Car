import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_and_distort_image(img,nx,ny):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Array to store object points and image points from all the image
    objpoints = []
    imgpoints = []

    objp = np.zeros((6*8,3),np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) #x,y coordinate

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)


    # If found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undist_image = cv2.undistort(img, mtx, dist, None, mtx)

        return mtx,dist,undist_image

def show_undistorted_image(img, undist_image):
    #show origninal image and distorted image
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(undist_image)
    ax2.set_title('Undistorted Image')
    plt.show()


# prepare object points
nx = 8
ny = 6
# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# get instrinct matrics, distortion indexes and undistored image 
mtx,dist,undist_image = calibrate_and_distort_image(img,nx,ny)
show_undistorted_image(img, undist_image)