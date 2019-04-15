import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
### %matplotlib inline
import glob
### %matplotlib qt
import pickle
import os


def createFolder(directory):
    for folder in directory:
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError:
            print('Error: Creating directory. ' + folder)

corners_found_folder = 'camera_cal/chessbooard_corners_found/'
corners_not_found_folder = 'camera_cal/chessbooard_corners_NOT_found/'

folderlist = []
folderlist.append(corners_found_folder)
folderlist.append(corners_not_found_folder)

def saveimage(img_path, ret, image):

    if ret == True:
        found = "_corners_found"
        save_folder = corners_found_folder
    elif ret == False:
        found = "_corners_NOT_found"
        save_folder = corners_not_found_folder

    foldername, filename = os.path.split(img_path)
    input_img_name, file_extension = os.path.splitext(filename)
    output_filename = input_img_name + found + file_extension
    #mpimg.imsave(save_folder + output_filename, image)
    cv2.imwrite(save_folder + output_filename, image)

# Read in calibration images
image_names = glob.glob('camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, line (0,0,0), (1,0,0), ....., (8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

createFolder(folderlist)

for image_name in image_names:
    img = mpimg.imread(image_name)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray,(9,6), None)
    #If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # draw and display the corners
        img = cv2.drawChessboardCorners(img,(9,6), corners, ret)
        saveimage(image_name, ret, img)
    elif ret == False:
        print('Corners Not Found For The Image {}'.format(image_name))
        saveimage(image_name, ret, img)

testImg = mpimg.imread('test_images/straight_lines1.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, testImg.shape[1::-1], None, None)

# Save the camera calibration result for later use
camera_params = {}
camera_params["objpoints"] = objpoints
camera_params["imgpoints"] = imgpoints
camera_params["ret"] = ret
camera_params["mtx"] = mtx
camera_params["dist"] = dist
camera_params["rvecs"] = rvecs
camera_params["tvecs"] = tvecs

pickle.dump(camera_params, open("camera_cal/camera_cal.p","wb"))
