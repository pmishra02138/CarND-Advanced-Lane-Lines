import pickle
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Undistort and apply perspective the image
def corners_unwarp(img, cal_file):

    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open( cal_file, "rb" ))
    mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]

    # Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    if (undist.ndim == 3):
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    else:
        gray = undist
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # Source points
    src = np.float32(
        [[288, 664],
         [1020, 664],
         [555, 478],
         [735, 478]]
    )

    # Destination points
    dst = np.float32(
        [[288, 664],
         [1020, 664],
         [288, 150],
         [975, 150]]
    )

    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, src

if __name__ == '__main__':

    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open( "pickle_data/camera_cal.p", "rb" ) )
    mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]

    # Read in an image
    img = cv2.imread('test_images/straight_lines1.jpg')

    #top_down, perspective_M, src = corners_unwarp(img, mtx, dist)
    warped, perspective_M, src = corners_unwarp(img, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    # Draw polygon
    pts = np.array([src[0], src[2], src[3], src[1]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(255,0,0),4)

    # Original image (with polygon overlapped)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=30)

    # Warped image
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted and Warped Image', fontsize=30)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
