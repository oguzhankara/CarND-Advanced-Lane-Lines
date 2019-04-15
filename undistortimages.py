import pickle
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def undistort_image(img, dist, mtx):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def saveimage(img_path, image, save_folder):

    foldername, filename = os.path.split(img_path)
    input_img_name, file_extension = os.path.splitext(filename)
    output_filename = input_img_name + "_undistorted" + file_extension
    #mpimg.imsave(save_folder + output_filename, image)
    cv2.imwrite(save_folder + output_filename, image)


def createFolder(directory):

    for folder in directory:
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError:
            print('Error: Creating directory. ' + folder)

# Read in the saved objpoints and imgpoints
camera_params = pickle.load( open( "camera_cal/camera_cal.p", "rb" ) )
objpoints = camera_params["objpoints"]
imgpoints = camera_params["imgpoints"]
ret = camera_params["ret"]
mtx = camera_params["mtx"]
dist = camera_params["dist"]
rvecs = camera_params["rvecs"]
tvecs = camera_params["tvecs"]


folderlist = []
chessboard_img_folder = 'camera_cal/undistorted_output/'

folderlist.append(chessboard_img_folder)

createFolder(folderlist)

chessboard_images = glob.glob('camera_cal/calibration*.jpg')

for image in chessboard_images:
    img = cv2.imread(image)
    undist_img = undistort_image(img, dist, mtx)
    saveimage(image, undist_img, chessboard_img_folder)


test_images = glob.glob('test_images/*.jpg')
test_img_output_folder = 'output_test_images/'
for image in test_images:
    img = cv2.imread(image)
    undist_img = undistort_image(img, dist, mtx)
    saveimage(image, undist_img, test_img_output_folder)