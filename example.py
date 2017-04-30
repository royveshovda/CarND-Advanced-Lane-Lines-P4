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
    img = cv2.imread("test_images/test3.jpg")
    img_dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite("output_images/undistorted.jpg", img_dst)


def filter_example():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("test_images/test3.jpg")
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_binary = process.filter_pipeline_single_image(img_undist)
    img_dst = cv2.cvtColor(img_binary*255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("output_images/filtered.jpg", img_dst)


def warped_example():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("test_images/test3.jpg")
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    M_persp, Minv_persp = process.get_perspective_transform_matrixes()
    img_size = (img.shape[1], img.shape[0])
    img_warped = cv2.warpPerspective(img_undist, M_persp, img_size, flags=cv2.INTER_LINEAR)
    cv2.imwrite("output_images/warped.jpg", img_warped)


def fit_lines_example():
    mtx, dist = process.calculate_camera_distortion()
    img = cv2.imread("test_images/test3.jpg")
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_filtered = process.filter_pipeline_single_image(img_undist)
    img_size = (img.shape[1], img.shape[0])
    M_persp, Minv_persp = process.get_perspective_transform_matrixes()
    img_warped = cv2.warpPerspective(img_filtered, M_persp, img_size, flags=cv2.INTER_LINEAR)
    new_left_fit, new_right_fit, img_out = process.fit_lines(img_warped)
    cv2.imwrite("output_images/lines.jpg", img_out)


def process_image_example():
    img = cv2.imread("test_images/test3.jpg")
    mtx, dist = process.calculate_camera_distortion()
    M_persp, Minv_persp = process.get_perspective_transform_matrixes()
    img_processed = process.process_image(img, mtx, dist, M_persp, Minv_persp)
    cv2.imwrite("output_images/processed.jpg", img_processed)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    undistort_example()
    filter_example()
    warped_example()
    fit_lines_example()
    process_image_example()
    logging.info('Finished')

if __name__ == '__main__':
    main()
