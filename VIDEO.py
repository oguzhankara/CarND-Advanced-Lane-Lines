import cv2
import matplotlib.image as mpimg
import pickle
from thresholding import combined_thresholding
from perspectivetransform import warper
from lanedetection import find_lane_pixels, search_around_poly, measure_curvature, vehicle_offset, drawing
from Line import Line
from moviepy.editor import VideoFileClip
import glob
import os


camera_params = pickle.load( open( "camera_cal/camera_cal.p", "rb" ) )
objpoints = camera_params["objpoints"]
imgpoints = camera_params["imgpoints"]
ret = camera_params["ret"]
mtx = camera_params["mtx"]
dist = camera_params["dist"]
rvecs = camera_params["rvecs"]
tvecs = camera_params["tvecs"]

lanes_detected = False

# Create line objects with Line class
right = Line()
left = Line()

def frameoperation(frame):

    global mtx, dist, detection, right, left, lanes_detected

    undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
    comb_thresh = combined_thresholding(undist_frame)
    warped, unwarped, M, M_Inv = warper(comb_thresh)

    # Sliding window algorithm will be used for "NOT" detected frames
    if not lanes_detected:
        """
        If this is the first frame or if the lane pixel positions cannot be detected with
        'Search Around Poly' algorithm; then 'Sliding Windows' algorithm will be used.
        
        `leftx`, `lefty`, `rightx`, `righty` are pixel positions within sliding windows
        
        `left_fit` and `right_fit` are filter coefficients of polynomials
        """
        leftx, lefty, rightx, righty, left_fit, right_fit, out_img = find_lane_pixels(warped)

        # Curvature values in meters (from pixel to real world calculations)
        left_curve, right_curve = measure_curvature(leftx, lefty, rightx, righty)
        left.update_coefficients(left_fit)
        right.update_coefficients(right_fit)

        lanes_detected = True

        print("##################    FIRST FRAME OR NOT DETECTED  ##################")

    else:
        """
        This is the part where we know lane lines from previous frames have been calculated
        To make the search for the new lana lines faster;
        'Seerch From Ploy' algorithm will be used
        
        `left_fit` and `right_fit` will be fetched from average poly filter coefficients and
        will be used to search around for the pixels 
        """
        # print("left_fetch", left.fetch_poly_coeff())
        left_fit = left.fetch_poly_coeff()
        right_fit = right.fetch_poly_coeff()

        left_fit_updated, right_fit_updated, confidence, leftx, lefty, rightx, righty = \
            search_around_poly(warped, left_fit, right_fit)

        if confidence:
            left_fit, right_fit = left_fit_updated, right_fit_updated
            left.update_coefficients(left_fit)
            right.update_coefficients(right_fit)
            left_curve, right_curve = measure_curvature(leftx, lefty, rightx, righty)

        else:
            lanes_detected = False

    v_offset = vehicle_offset(undist_frame, left_fit, right_fit)

    # Returned image will be visualizations on top of undistorted image.
    final = drawing(undist_frame, left_fit, right_fit, M_Inv, left_curve, right_curve, v_offset)

    return final

def annotation(input_file, output_file):
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(frameoperation)
    annotated_video.write_videofile(output_file, audio=False)




if __name__ == '__main__':

    annotation('project_video.mp4', 'project_video_output.mp4')

    """
    def saveimage(img_path, image, save_folder):

        foldername, filename = os.path.split(img_path)
        input_img_name, file_extension = os.path.splitext(filename)
        output_filename = input_img_name + "_annotated" + file_extension
        mpimg.imsave(save_folder + output_filename, image, cmap='gray')

    test_images = glob.glob('output_test_images/*_undistorted.jpg')
    test_img_output_folder = 'output_test_images/'
    for image in test_images:
        frame = mpimg.imread(image)
        annotated_frame = frameoperation(frame)
        saveimage(image, annotated_frame, test_img_output_folder)
    """