import process
import cv2
import logging


def calibrate():
    mtx, dist = process.calculate_camera_distortion()
    print(mtx)


def hls_select(img, thresh=(230, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


def abs_sobel_x(img, thresh=(20, 100)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def filter():
    img = cv2.imread("test_images/test1.jpg")
    img2 = process.filter_pipeline_single_image(img)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    calibrate()
    logging.info('Finished')

if __name__ == '__main__':
    main()