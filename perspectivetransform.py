import cv2
import numpy as np
import glob
import os
import matplotlib.image as mpimg

# TODO: Final SRC and DST points should be optimized
def warper(img):

    # Source and destination points were calculated before.
    src = np.float32([[218., 702.],[564., 468.],[716., 468],[1087., 702.]])
    dst = np.float32([[320., 702.],[320., 0.],[960., 0.],[980., 702.]])

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_Inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    warped_size = (warped.shape[1], warped.shape[0])
    unwarped = cv2.warpPerspective(warped, M_Inv, warped_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    # TODO: return updated from warped, unwarped to warped, unwarped, M, M_Inv
    return warped, unwarped, M, M_Inv




if __name__ == '__main__':


    def saveimage(img_path, image, save_folder):

        foldername, filename = os.path.split(img_path)
        input_img_name, file_extension = os.path.splitext(filename)
        output_filename = input_img_name + "_warped" + file_extension
        mpimg.imsave(save_folder + output_filename, image, cmap='gray')

    test_images = glob.glob('output_test_images/*undistorted*.jpg')
    
    test_img_output_folder = 'output_test_images/'
    for image in test_images:
        img = mpimg.imread(image)
        warped_img, unwarped_img, M_temp, M_Inv_temp = warper(img)
        saveimage(image, warped_img, test_img_output_folder)
