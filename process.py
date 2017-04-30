import numpy as np
import cv2
import glob
import logging


def calculate_camera_distortion():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('camera_cal/calibration*.jpg')
    logging.debug(images)

    # Step through the list and search for chessboard corners
    for filename in images:
        img = cv2.imread(filename)
        logging.debug(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img = cv2.imread("camera_cal/calibration2.jpg")
    img_size = (img.shape[1], img.shape[0])
    _ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist


def filter_pipeline_single_image(img, s_thresh=(185, 255), sx_thresh=(40, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hsv[:, :, 2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Derivative in x
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[
                    ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))
                    |
                    ((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))
                ] = 1
    return binary_output


def get_perspective_transform_matrixes():
    src = np.float32([[585, 460], [696, 460], [1127, 720], [203, 720]])
    dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

    M_persp = cv2.getPerspectiveTransform(src, dst)
    Minv_persp = cv2.getPerspectiveTransform(dst, src)

    return M_persp, Minv_persp


def fit_lines(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image)) * 255

    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(image.shape[0] / nwindows)
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit = []
    else:
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = []
    else:
        right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, out_img


def curvature(left_fit, right_fit, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])
    center = (((left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2]) + (
    right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2])) / 2 - 640) * xm_per_pix

    return left_curverad, right_curverad, center


def draw_line(undist, warped, left_fit, right_fit, Minv_persp):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    # Fit new polynomials to x,y in world space
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (255, 215, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv_persp, (color_warp.shape[1], color_warp.shape[0]))

    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result, color_warp


global left_fit
global right_fit
left_fit = None
right_fit = None


def process_image(img, mtx, dist, M_persp, Minv_persp, s_thresh=(180, 255), sx_thresh=(40, 100)):
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_filtered = filter_pipeline_single_image(img_undist, s_thresh, sx_thresh)
    img_size = (img.shape[1], img.shape[0])
    img_warped = cv2.warpPerspective(img_filtered, M_persp, img_size, flags=cv2.INTER_LINEAR)

    global left_fit
    global right_fit

    new_left_fit, new_right_fit, _ = fit_lines(img_warped)
    if left_fit is None:
        left_fit = new_left_fit
    else:
        if new_left_fit is not []:
            left_fit[0] = 0.9 * left_fit[0] + 0.1 * new_left_fit[0]
            left_fit[1] = 0.9 * left_fit[1] + 0.1 * new_left_fit[1]
            left_fit[2] = 0.9 * left_fit[2] + 0.1 * new_left_fit[2]

    if right_fit is None:
        right_fit = new_right_fit
    else:
        if new_right_fit is not []:
            right_fit[0] = 0.9 * right_fit[0] + 0.1 * new_right_fit[0]
            right_fit[1] = 0.9 * right_fit[1] + 0.1 * new_right_fit[1]
            right_fit[2] = 0.9 * right_fit[2] + 0.1 * new_right_fit[2]

    left_curv, right_curv, center_off = curvature(left_fit, right_fit, img_warped)

    # Warp back to original and merge with image
    img_out, img_birds = draw_line(img_undist, img_warped, left_fit, right_fit, Minv_persp)

    # Write curvature and center in image
    text_left = "Left curv: " + str(int(left_curv)) + " m"
    text_right = "Right curv: " + str(int(right_curv)) + " m"
    text_center = "Center offset: " + str(round(center_off, 2)) + "m"
    font_scale = 1
    thickness = 2

    font_face = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img_out, text_left, (500, 40), font_face, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, text_right, (500, 70), font_face, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, text_center, (500, 100), font_face, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return img_out


def process_project_video():
    from moviepy.editor import VideoFileClip
    project_output = "project.mp4"

    mtx, dist = calculate_camera_distortion()
    M_persp, Minv_persp = get_perspective_transform_matrixes()
    clip_project = VideoFileClip("project_video.mp4")
    pro = lambda img: process_image(img, mtx, dist, M_persp, Minv_persp)
    project_clip = clip_project.fl_image(pro)
    project_clip.write_videofile(project_output, audio=False)


def process_challenge_video():
    from moviepy.editor import VideoFileClip
    project_output = "out_challenge.mp4"

    mtx, dist = calculate_camera_distortion()
    M_persp, Minv_persp = get_perspective_transform_matrixes()
    clip_project = VideoFileClip("challenge_video.mp4")
    pro = lambda img: process_image(img, mtx, dist, M_persp, Minv_persp)
    project_clip = clip_project.fl_image(pro)
    project_clip.write_videofile(project_output, audio=False)


def process_harder_video():
    from moviepy.editor import VideoFileClip
    project_output = "out_harder_project.mp4"

    mtx, dist = calculate_camera_distortion()
    M_persp, Minv_persp = get_perspective_transform_matrixes()
    clip_project = VideoFileClip("harder_challenge_video.mp4")
    pro = lambda img: process_image(img, mtx, dist, M_persp, Minv_persp)
    project_clip = clip_project.fl_image(pro)
    project_clip.write_videofile(project_output, audio=False)
