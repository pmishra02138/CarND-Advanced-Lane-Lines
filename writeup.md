## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/undistorted_road.png "Road Transformed"
[image3]: ./output_images/binary_combo.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/highlight_lane.png "Output"
[video1]: ./project_video.mp4 "Video"

### The individual steps of this implementation are described below.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `camera_cal.py`.  I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

Camera matrix and distortion coefficients are stored in `camera_cal.p` in `pickle_data` folder.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied the distortion correction to one of the test images: `test_images/straight_lines1.jpg`.

![alt text][image2]

The undistorted image is in the `output_images/undistorted_road.png`.
The code to execute this is shown in the `camera_cal.py` file in lines from 56-68.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. This function is implemented in `color_pipeline(...)` function of `color_transform.py` file. In the first step calculate Sobel transform for x gradient only (line # 14-22).  Therefater, HLS transform of of the color image is calculated in line #24-26.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp(...)`, which appears in lines 12 through 52 in the file `perspective_transform.py`.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane-line pixels identification code is writtrn from lines #19-91 in function `sliding_window(...)` in `detect_lanes.py` file. This function accepts a binary image calculated using color transform. I have sliding windows technique for lane detection. One such image is as follows:

![alt text][image5]

(Note: Please uncomment `detect_lanes_test_images()` line in main function to run lane-pixel detection on every test image.)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # 93 through # 119 in my code in  `calculate_curvature` function in file `detect_lanes.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # 135 through #157 in my code in `highlight_lanes(...)` function `detect_lanes.py`file.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The lane highlight depends on the edge detection and curvature calculations. In case of incorrect edge detection vehicle can run off the road. To handle this, I have use distance between lanes vehicle center offset from camera to detect any anomaly (line # 210). In case a bad edge is detected, I use values from pervious detection. This works for this case but taking a running average would be a better solution.
