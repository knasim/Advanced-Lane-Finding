## Advanced Lane Finding

---

The goals / steps of this effort are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[video2]: ./output.mp4 "Video Output Result"
[image7]: ./output_images/chessboard-original-undistorted.png "Chessboard Undistorted"
[image8]: ./output_images/source_highway_original_undistorted.png "Highway Undistorted"
[image9]: ./output_images/source_highway_thresholded_magnitude.png "Thresholded Magnitude"
[image10]: ./output_images/source_highway_undistorted_warped.png "Warped"
[image11]: ./output_images/polynomial.png "Second Order Polynomial"
[image12]: ./output_images/lane-lines.png "Lane Lines Identified"


### The Code
The main program is in the file befittingly named `main.py`.   It can be executed from the terminal.  
There is another file `function.py` that has supporting functions used by the main program.

### Camera Calibration

The code for this step is contained in (main.py lines 30-38).

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world (main.py lines 21-22).
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image7]



### Pipeline (single images)

#### 1. The following is an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image8]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To achieve a thresholded binary image I experimented with various methods refer to (function.py lines 10-86).
A combination of absolute, magnitude, direction methods were applied which resulted in desired output.
To futher improve on this, LUV/Lab combinations was tried.  The B and L channels were the best choice
to lighting variance on sections of the road.  Refer to (function.py) for details.
The resulting ouptput was:

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 89 through 112 in the file `function.py`.  I used the cv2.warpPerspective function to achieve this refer to (function.py lines 89-112)
The challenge was a trail and error to find the correct values for `src` and `dst` (function.py lines 103,105).  
See the resultant image below:

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? (lane line pixels)

To identify lane lines see (main.py lines 160-217) mostly code reused from relevant lectures.  I applied second order polynomial for this.

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is computed using the average of the left and right radii.
The distance between the vehicle and the center is a function of the average of the lanes relative to the center.
The values of the radius and position are displayed on the video.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result showing identified lanes:


![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a direct link to YouTube for the result on the entire project video:  [Advanced lane finding video output](https://youtu.be/4kHg892OFAo)

Here's a [direct link to my video file](./output.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.

Challenges Faced:
  1. Proper thresholding to identify the lanes.

Improvements:
  1. Variance in road lighting can give problems.  There is room to improve on this.

Potential Problem ?
  1.  Velocity of the car be a factor in the rendering of the video.  It would be worth exploring how efficient
      the program will work as the rate of the vehicles velocity increases.
