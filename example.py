import process
import cv2
import logging


def calibrate():
    mtx, dist = process.calculate_camera_distortion()
    print(mtx)


def filter():
    img = cv2.imread("test_images/test1.jpg")
    img2 = process.filter_pipeline_single_image(img)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    process.process_project_video()
    logging.info('Finished')

if __name__ == '__main__':
    main()
