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

def hls_select(img, thresh=(230, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


def white_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    w_h_thresh = [4, 255]
    w_s_thresh = [0, 32]
    w_v_thresh = [207, 255]
    binary_output = np.zeros_like(H)
    binary_output[(H >= w_h_thresh[0]) & (H <= w_h_thresh[1]) & (S >= w_s_thresh[0]) & (S <= w_s_thresh[1]) & (V >= w_v_thresh[0]) & (V <= w_v_thresh[1])] = 1
    return binary_output


def yellow_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    y_h_thresh = [18, 80]
    y_s_thresh = [80, 255]
    y_v_thresh = [0, 255]
    binary_output = np.zeros_like(H)
    binary_output[(H >= y_h_thresh[0]) & (H <= y_h_thresh[1]) & (S >= y_s_thresh[0]) & (S <= y_s_thresh[1]) & (V >= y_v_thresh[0]) & (V <= y_v_thresh[1])] = 1
    return binary_output


def white_yellow_lines(img):
    yellow_lane = yellow_lines(img)
    white_lane = white_lines(img)
    binary_output = cv2.bitwise_or(yellow_lane, white_lane)
    return binary_output


def white_example():
    img = cv2.imread("test_images/test3.jpg")
    img_binary = process.color_filter(img, h_thresh=(4, 255), s_thresh=(0, 32), v_thresh=(207, 255))
    img_dst = cv2.cvtColor(img_binary * 255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("output_images/white.jpg", img_dst)


def yellow_example():
    img = cv2.imread("test_images/test3.jpg")
    img_binary = process.color_filter(img, h_thresh=(18, 110), s_thresh=(100, 255), v_thresh=(100, 255))
    img_dst = cv2.cvtColor(img_binary * 255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("output_images/yellow.jpg", img_dst)


def white_and_yellow_example():
    img = cv2.imread("test_images/test3.jpg")
    img_binary = white_yellow_lines(img)
    img_dst = cv2.cvtColor(img_binary * 255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("output_images/white_yellow.jpg", img_dst)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    #undistort_example()
    #filter_example()
    #warped_example()
    #fit_lines_example()
    #process_image_example()
    #white_example()
    #yellow_example()
    #white_and_yellow_example()
    # process_image_example()
    process.process_project_video()
    #process.process_challenge_video()
    #process.process_harder_video()
    logging.info('Finished')



if __name__ == '__main__':
    main()
