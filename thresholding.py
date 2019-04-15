import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Run the function
# grad_binary = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
# mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
# dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
# hls_binary = hls_select(image, thresh=(90, 255))

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Absolute Sobel Threshold Function
    Calculates directional gradient
    Takes an image, gradient orientation, and threshold min / max values.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Magnitude of the gradient function.
    Calculates gradient magnitude
    Aplies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image

    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Direction of the gradient function
    Calculates gradient direction
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return dir_binary

def hls_select(img, thresh=(0, 255)):
    """
    Obtain S channel from the image by converting from RGB to HLS color space
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return hls_binary

def combined_thresholding(img):
    """
    :return: Combination of thresholding abs_sobel_thresh(), mag_thresh(), dir_threshold(), hls_select()
    """
    grad_x_bin = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
    grad_y_bin = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))
    dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_bin = hls_select(img, thresh=(170, 255))
    combined_bin = np.zeros_like(dir_bin)
    #combined_bin[(((grad_x_bin == 1) | (grad_y_bin == 1)) | ((mag_bin == 1) & (dir_bin == 1))) | (hls_bin == 1)] = 1
    combined_bin[(grad_x_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1
    #combined_bin[(grad_x_bin == 1) & (grad_y_bin == 1)] = 1

    return combined_bin



if __name__ == '__main__':


    def saveimage(img_path, image, save_folder):

        foldername, filename = os.path.split(img_path)
        input_img_name, file_extension = os.path.splitext(filename)
        output_filename = input_img_name + "_thresholded" + file_extension
        mpimg.imsave(save_folder + output_filename, image, cmap='gray')

    test_images = glob.glob('output_test_images/*_undistorted.jpg')
    test_img_output_folder = 'output_test_images/'
    for image in test_images:
        img = mpimg.imread(image)
        undist_img = combined_thresholding(img)
        saveimage(image, undist_img, test_img_output_folder)