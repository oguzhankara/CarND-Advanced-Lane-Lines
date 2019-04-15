# Advanced Lane Finding Project

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Creation of object and image point arrays can be found in `cameracalibration.py` as the following;
```python
  # Arrays to store object points and image points from all the images
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane

  # Prepare object points, line (0,0,0), (1,0,0), ....., (8,5,0)
  objp = np.zeros((6*9,3), np.float32)
  objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates
```

All the chessboard calibration images are fetched with glob API. Corners are found with the help of OpenCV function findChessboardCorners().

```python
  import glob
  image_names = glob.glob('camera_cal/calibration*.jpg')
  ..
  ...
  ret, corners = cv2.findChessboardCorners(gray,(9,6), None)
  if ret == True:
      imgpoints.append(corners)
      objpoints.append(objp)
```
![alt text](camera_cal\chessbooard_corners_found\calibration17_corners_found.jpg)

**Note**: In this process, it is found that for 3 sample calibration images, corners couldn't be detected.

```python
  Corners Not Found For The Image camera_cal\calibration1.jpg
  Corners Not Found For The Image camera_cal\calibration4.jpg
  Corners Not Found For The Image camera_cal\calibration5.jpg
```

After, objpoints and imgpoints lists are populated, they are used together with one of the test images by using OpenCV function `calibrateCamera()` to calculate the camera calibration and distortion coefficients. These coefficients are saved in a pickle named **camera_cal.p** and saved under **./camera_cal** directory for later use of undistortion.
```python
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, testImg.shape[1::-1], None, None)
  import pickle
  pickle.dump(camera_params, open( "camera_cal/camera_cal.p", "wb" ) )
```
Here, we are ready to undistort the image by using OpenCV function `undistort(img, mtx, dist, None, mtx)`. Details can be found in `undistortimages.py`. Following is the **distorted** and **undistorted** image comparison.

<p align="top">
  <img align="left" img src="/camera_cal/calibration1.jpg" width="400" />  <img align="right" img src="/camera_cal/undistorted_output/calibration1_undistorted.jpg" width="400" />
</p>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

With the same techniques explained above, test images are undistorted by using `undistortimages.py`. Undistorted images are saved under **/output_test_images** directory. Following is the **distorted** and **undistorted** image comparison for _test1.jpg_ provided in **/test_images** directory.

<p align="top">
  <img align="left" img src="/test_images/test1.jpg" width="400" />  <img align="right" img src="/output_test_images/test1_undistorted.jpg" width="400" />
</p>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>





#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a thresholded binary image. The whole thresholding operation is done in `thresholding.py` with the following functions below. Explanations of all the thresholing techniques are stated in the functions. The function structures are from the lectures.

```python
  def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
      """
      Absolute Sobel Threshold Function
      Calculates directional gradient
      Takes an image, gradient orientation, and threshold min / max values.
      -> thresh=(20, 100)
      """

  def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
      """
      Magnitude of the gradient function.
      Calculates gradient magnitude
      Aplies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
      -> mag_thresh=(30, 100)
      """

  def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
      """
      Direction of the gradient function
      Calculates gradient direction
      -> thresh=(0.7, 1.3)
      """

  def hls_select(img, thresh=(0, 255)):
      """
      Obtain S channel from the image by converting from RGB to HLS color space
      -> thresh=(170, 255)
      """

  def combined_thresholding(img):
      """
      :return: Combination of thresholding abs_sobel_thresh(), mag_thresh(), dir_threshold(), hls_select()
      """
```
Here's an example of my output for this step. Following is the binary thresholded image example from test images. Other sample outputs can be find in **/output_test_images** directory. Here, we are not worried about the noise resides in the image. These will be filtered out in the perspective transform and lane detection phases. Overall, we obtain the lines clearly.

<p align="top">
  <img align="left" img src="/output_test_images/straight_lines2_undistorted.jpg" width="400" />  <img align="right" img src="/output_test_images/straight_lines2_undistorted_thresholded.jpg" width="400" />
</p>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>




#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 8 through 24 in the file `perspectivetransform.py`. The `warper()` function takes as inputs an image (`img`). I chose to hardcode the source and destination points with eyeball estimate. It would be smart to select these points automatically.

```python
  src = np.float32([[218., 702.],[564., 468.],[716., 468],[1087., 702.]])
  dst = np.float32([[320., 702.],[320., 0.],[960., 0.],[980., 702.]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 218, 702      | 320, 702      |
| 564, 468      | 320, 0        |
| 716, 468      | 960, 0        |
| 1087, 702     | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


<p align="center">
  <img align="left" img src="/output_test_images/straight_lines1_undistorted.jpg" width="250" /> <img align="center" img src="/output_test_images/straight_lines1_undistorted_warped.jpg" width="250" /> <img align="right" img src="/output_test_images/straight_lines1_undistorted_thresholded_warped.jpg" width="250" />
</p>
<br/>

Additionally, I tried another approach to decide on the points by creating polygon lines on the image, however I found it hard to fine tune it since the coordinates was calculated as proportional to image shape. I did not add it to below for brevity.
```python
  def draw_polylines(img, poly_lines):
    poly_lines = poly_lines.reshape((-1, 1, 2))
    source_lines_drawed = cv2.polylines(img, [poly_lines], True, (255, 0, 0), thickness=2)
    return source_lines_drawed

    polygon_lines = np.array([[left_bottom_coor, left_top_coor, right_top_coor, right_bottom_coor]], dtype=np.int32)
    lines_drawed = draw_polylines(img, polygon_lines)
```


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I took a histogram of the bottom half of the **binary frame**. Then, found the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. Then, I defined some hyper-parameters as the following.

| Hyper-Parameter| Destination |
|:--------------:|:-------------:|
| `nwindows = 9`   | The number of _sliding windows_ |
| `margin = 100`   | Width of the windows +/- margin |
| `minpix = 50`    | Minimum number of pixels found to recenter window |
| `window_height`  | Based on frame _height / nwindows_ |


After that, I identified the x and y positions of all nonzero pixels in the image as the following.
```python
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
```
I kept current positions to be updated later for each window in _sliding windows_, created empty lists to receive left and right lane pixel indices and stepped through the windows one by one. By doing that, nonzero pixels are determined for x and y coordinates within the window. Then, previously created empty lists are populated. If the pixels found in the window is greater then 50 (`minpix`), then the sliding window is re-centered based on the pixels' mean positions. After, extracting left and right line pixel positions, a second order polynomial is applied with the following code snippet.
```python
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
```
The code for the operations stated above can be seen in `find_lane_pixels()` function which appears in lines 7 through 90 in the file `lanedetection.py`



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

##### _Radius of Curvature_
I am assuming that projected section of the lane is about 30 meters long (which corresponds to ~720 pixels) and 3.7 meters wide (which corresponds tp ~700 pixels). Following code snippet in `measure_curvature()` function in `lanedetection.py` file shows how I make the conversion from pixels to real-world.
```python
  xm_per_pix = 3.7/700 # meters per pixel in x dimension
  ym_per_pix = 30/720 # meters per pixel in y dimension
```

As shown in the lectures, the radius of curvature at any point x of the function _x = f(y)_ is the following;

![alt text](formula\Radius_of_curvature_1.png)

For the second order polynomials, below shows the first and second derivatives;

![alt text](formula\Radius_of_curvature_2.png)

So, the equation for radius of curvature becomes the following;

![alt text](formula\Radius_of_curvature_3.png)

Therefore, this formula can be represented in python like the following. This cone snippet can be seen inside `measure_curvature()` function in `lanedetection.py` file.

```python
# The calculation of R_curve (radius of curvature) in meters
  left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    2 * left_fit_cr[0])
  right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    2 * right_fit_cr[0])
```
##### _Vehicle Offset_

Vehicle's center is assumed to be the center of the image. Also, lanes' center is calculated to be the bottom midpoint of mean x value of both of the lanes. The function to calculate the vehicle's offset is `vehicle_offset()` which can be found in `lanedetection.py` file.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Following is the final output of one of the annotated test images. Other annotations for remaining test images can be found in **/output_test_images** directory. To run the pipeline on the test images, you can use `VIDEO.py` which is in the main directory.

![alt text](output_test_images\test6_undistorted_annotated.jpg)


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Video annotation is handled with `drawing()` function which can be found in `lanedetection.py` file. Steps applied are the following;
  * Generate x and y values for plotting
  * Create an image to draw the lines on
  * Recast the x and y points into usable format for cv2.fillPoly()
  * Draw the lane onto the warped blank image
  * Warp the blank back to original image space using inverse perspective matrix (Minv)
  * Combine the result with the original image
  * Annotate lane curvature values and vehicle offset from center

To run the pipeline, you can use `VIDEO.py` which is in the main directory.
Annotated video can be found in the main project directory with the name **challenge_video.mp4**

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In overall, I am satisfied with the results.

But, there are some points I would like to stress.

The pipeline is actually tried also with challenge video. However, the result is not shared. I knew that it would not work properly since the pipeline is not precise to differentiate between the long cracks in the pavement and the actual lane lines.

Moreover, even tough several thresholding techniques or color channel filtering are applied; the algorithm still needs more effort to handle sudden color or shadow changes on the pavement. Actually, some smoothing techniques tried to be applied with using **Line Class** (can be seen in `Line.py`), but still it is not that robust. I am open to new ideas.

Following is one of the most catastrophic case that I've faced through the video feed.

![alt text](output_test_images\detection_failure.png)
