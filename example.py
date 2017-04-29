import process
import cv2
import logging


def calibrate():
    mtx, dist = process.calculate_camera_distortion()
    print(mtx)


def filter():
    img = cv2.imread("test_images/test1.jpg")
    img2 = process.filter_pipeline_single_image(img)


def process_video():
    from moviepy.editor import VideoFileClip
    project_output = "project.mp4"

    mtx, dist = process.calculate_camera_distortion()
    M_persp, Minv_persp = process.get_perspective_transform_matrixes()
    #clip_project = VideoFileClip("project_video.mp4").subclip(40.8, 42)
    clip_project = VideoFileClip("project_video.mp4")
    pro = lambda img: process.process_image(img, mtx, dist, M_persp, Minv_persp)
    project_clip = clip_project.fl_image(pro)
    project_clip.write_videofile(project_output, audio=False)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Started')
    process_video()
    logging.info('Finished')

if __name__ == '__main__':
    main()