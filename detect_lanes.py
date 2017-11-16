import pickle
import numpy as np
import cv2
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from perspective_transform import corners_unwarp
from color_transform import color_pipeline

# Constants for sliding windows
NWINDOWS = 9 # Number of sliding windows
MARGIN = 100 # Width of the windows +/- margin
MINPIX = 50 # Minimum number of pixels found to recenter window

def sliding_window(binary_warped):
    # Take a histogram of the bottom half of the warped binary image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/NWINDOWS)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(NWINDOWS):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - MARGIN
        win_xleft_high = leftx_current + MARGIN
        win_xright_low = rightx_current - MARGIN
        win_xright_high = rightx_current + MARGIN

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > MINPIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MINPIX:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, ploty, left_fitx, right_fitx

def calculate_curvature(ploty, left_fitx, right_fitx, img_shape):

    # Define y-value where we want radius of curvature
    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ykm_per_pix = 30/(720*1000) # kilometers per pixel in y dimension
    xkm_per_pix = 3.7/(700*1000) # kilometers per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ykm_per_pix, left_fitx*xkm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ykm_per_pix, right_fitx*xkm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ykm_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ykm_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate camera offset from center
    x_center = (img_shape[1]/2)*xkm_per_pix
    ymax = (img_shape[0])*ykm_per_pix
    left_pt = left_fit_cr[0]*ymax**2 + left_fit_cr[1]*ymax + left_fit_cr[2]
    right_pt = right_fit_cr[0]*ymax**2 + right_fit_cr[1]*ymax + right_fit_cr[2]
    lane_distance = abs(left_pt - right_pt)*1000
    center_offset = ((left_pt + right_pt)/2 - x_center) *1000

    return left_curverad, right_curverad, center_offset, lane_distance

def plot_lanes(out_img, ploty, left_fitx, right_fitx):

    plt.figure()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    # Display every plto for 2 secs
    plt.show(block=False)
    time.sleep(2)
    plt.close()

def highlight_lanes(img, binary_warped, Minv, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #
    # # Display every plto for 2 secs
    # plt.show(block=False)
    # time.sleep(2)
    # plt.close()

    return result

def detect_lanes_test_images():
        # Read in the saved camera matrix and distortion coefficients
        cal_file = "pickle_data/camera_cal.p"

        # Read in an image
        image_files = glob.glob('test_images/*.jpg')

        # Go through every file
        for image_file in image_files:
            img = cv2.imread(image_file)
            # Binary color transform
            color_binary, combined_binary = color_pipeline(img)

            # Undistort and apply perspective the image
            binary_warped, perspective_M, Minv, src = corners_unwarp(combined_binary, cal_file)

            # Sliding window for detecting lanes
            out_img, ploty, left_fitx, right_fitx = sliding_window(binary_warped)
            # Calculate radius of curvature
            left_rad, right_rad, center_offset, lane_distance = calculate_curvature(ploty, left_fitx, right_fitx, out_img.shape)
            # print(round(left_rad,2), 'km', round(right_rad, 2), 'km')

            # Visualize lanes and fitted polynomial
            plot_lanes(out_img, ploty, left_fitx, right_fitx)

            # Create an image to draw the lines on
            result = highlight_lanes(img, binary_warped, Minv, ploty, left_fitx, right_fitx)

        return result

Lane = {"left_fitx": None, "right_fitx": None, "left_rad": None, "right_rad": None, "center_offset": None}

def detect_lanes_video(img):

    # Binary color transform
    color_binary, combined_binary = color_pipeline(img)

    # Undistort and apply perspective the image
    cal_file = "pickle_data/camera_cal.p"
    binary_warped, perspective_M, Minv, src = corners_unwarp(combined_binary, cal_file)

    # Sliding window for detecting lanes
    out_img, ploty, left_fitx, right_fitx = sliding_window(binary_warped)

    # Calculate the lane curvatures and car's center offset
    left_rad, right_rad, center_offset, lane_distance = calculate_curvature(ploty, left_fitx, right_fitx, out_img.shape)

    # Sanity check for detected lanes
    if (lane_distance < 3.5) or (lane_distance > 4.2) or (abs(center_offset) > 0.5):
        left_fitx = Lane["left_fitx"]
        right_fitx = Lane["right_fitx"]
        left_rad = Lane["left_rad"]
        right_rad = Lane["right_rad"]
        center_offset = Lane["center_offset"]


    Lane["left_fitx"] = left_fitx
    Lane["right_fitx"] = right_fitx
    Lane["left_rad"] = left_rad
    Lane["right_rad"] = right_rad
    Lane["center_offset"] = center_offset

    # Highlight the area between lanes
    result = highlight_lanes(img, binary_warped, Minv, ploty, left_fitx, right_fitx)

    # Show left and right curvature and vechicle center offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    cv2.putText(result, 'Left curvature: {:.2f} km'.format(left_rad), (20, 50), font, 1, fontColor, 2)
    cv2.putText(result, 'Right curvature: {:.2f} km'.format(right_rad), (20, 100), font, 1, fontColor, 2)
    cv2.putText(result, 'Vehicle center offset {:.2f} m'.format(center_offset), (20, 150), font, 1, fontColor, 2)

    return result


if __name__ == '__main__':
    # Test lane detection in test images()
    # detect_lanes_test_images()
    from moviepy.editor import VideoFileClip

    output_video = 'output_images/lanes_output.mp4'
    clip1 = VideoFileClip("output_images/lanes.mp4")
    # clip1 = VideoFileClip("output_images/lanes.mp4").subclip(0, 10)
    clip = clip1.fl_image(detect_lanes_video)
    clip.write_videofile(output_video, audio=False)
