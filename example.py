import process
import cv2
import logging
import numpy as np


def undistort_camera_calibration():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("camera_cal/calibration2.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_images/calibration_undist.jpg", dst_bgr)


def undistort_example():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("test_images/test1.jpg")
    img_dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite("output_images/undistorted.jpg", img_dst)


def filter_example():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("test_images/test1.jpg")
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_binary = process.filter_pipeline_single_image(img_undist)
    img_dst = cv2.cvtColor(img_binary*255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("output_images/filtered.jpg", img_dst)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    #process.process_project_video()
    filter_example()
    logging.info('Finished')

if __name__ == '__main__':
    main()
