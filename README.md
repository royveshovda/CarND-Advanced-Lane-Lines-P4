## Project Writeup
This writeup is based on the provided template.

---

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

[image0]: ./camera_cal/calibration2.jpg "Uncalibrated"
[image1]: ./output_images/calibration_undist.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg "Road Transformed"
[image3]: ./output_images/filtered.jpg "Binary Example"
[image4]: ./output_images/warped.jpg "Warp Example"
[image5]: ./output_images/lines.jpg "Fit Visual"
[image6]: ./output_images/processed.jpg "Output"
[video1]: ./project.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I have provided this writeup as part of a GitHub-repo, so this is the README you are reading.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 32 of the file called 'process.py'.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in all but two images.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:
##### Original
![original][image0]
##### Undistorted
![undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
In the file named 'example.py' I have an example of how I undistort an image. In the function called 'undistort_example' (lines 15-19). I used the output from the camera calibration process, and call OpenCV's function 'cv2.undistort' to correct the image.

##### Example of undistorted image
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for the filter can be found in the file named 'process.py' in the method called 'filter_pipeline_single_image' (lines 37 through 55).

To filter an image I used a combination of threshold filter on the S-channel in the HLS-colorspace, and the x gradient in a Sobel filter (on greyscale converted image). The hard part was to tune the parameters to detect lanes under all light conditions. I spent a lot of time tuning these parameters.


##### Example of filtered image
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I have a function called 'get_perspective_transform_matrixes' (lines 58-65) in the file 'process.py'. This function return both transformation matrixes for normal to birds-eye view, and back again.

These matrixes are based on the following source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |


##### Example of warped image
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window algorithm, described in class to detect the left and right lines. This algorithm basically tried to detect to most likely area for where the line might be. For the base of the search I used a histogram output, splits this in the middle, and decides the highest spike in the left and right part will be the starting point for each line. After I have found these windows stacked on to of each other (as shown in the image below), I use Numpy's function 'polyfit', which gives me back second order function fitted to the detected windows. The detected lines are not drawn in the picture, but they always follow the center of the detected windows.

The code for this is located in 'process.py' in a function called 'fit_lines' (lines 68-130).

An example of usage is shown in 'example.py' in the function 'fit_lines_example' (lines 41-50).

It is important to note that thsi algorithm works on a filtered and warped image to make sense.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the center offset and curvature of the left and right lines based on the fitted lines from the previous step ('fit_lines'). The code is located in the file 'process.py' in the function 'curvature' (lines 133-147).


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


I extracted the part where I process an image and project back from birds-eye view in 'example.py' in a function called 'process_image_example' (lines 53-58).

The processing happens in 'process.py' in the functions 'process_image' (lines 179-223), and 'draw_line' (lines 150-170). The 'draw_line' function plots the left and right lines, and warpes the image back from birds-eye perspective, while the function 'process_image' smooths the fitted lines and applies the text for the curvatures and center offset.

The smoothing of the lines happens with a factor of 0.9 for the previous line, and 0.1 for the newly fitted line. This way the previous line is much more important than the newly detected line, and slows down the detected area from jumping around under harder conditions.

##### Example of processed image
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I decided to go for the suggestions from class and use only Sobel X and S-channel in HLS colorspace. This turned out to be very hard to tune correctly. The filters are very sensitive to markings on the road and color changes, like shadows. I also had to implement a smoothing algorithm to prevent the lines to jump around too much between each frame. As a result of this, the performance on the harder videos are rather poor. I do not have good enough filters for the 'challenge_video.mp4', and the smoothing part makes the algorithm adapt too slow to the turning road in 'harder_challenge_video.mp4' (filters should also be improved here).

So In summary the best place to improve will be to get better filters. I could also use previous findings in the window search, to avoid starting over each time. This would improve the left line a lot, as it jumps around a bit in the current implementation.
