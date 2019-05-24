from imutils.video import WebcamVideoStream, FileVideoStream
import cv2
import time
from detection import detect
from reid import re_id

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def re_id_process():
    re_id_model = re_id.Re_id()
    score = re_id_model.compare("test_data/query1.jpg", "test_data/query2.jpg")
    print(score)


def detect_process():
    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")

    # use camera
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)

    # read video
    video = FileVideoStream("test_data/test3.mp4").start()

    # start fps timer
    # loop over frames from the video file stream
    frame_count = 0
    detect_model = detect.Detect()

    while True:
        # grab next frame
        # frame = stream.read()
        frame = video.read()

        key = cv2.waitKey(1) & 0xFF

        if frame_count % 10 != 0:
            frame_count += 1
            continue
        frame_count += 1

        frame = detect_model.predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


def detect_with_reid_process():
    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")

    # use camera
    # stream = WebcamVideoStream(src=0).start()  # default camera
    # time.sleep(1.0)

    # read video
    video = FileVideoStream("test_data/test3.mp4").start()

    # start fps timer
    # loop over frames from the video file stream
    frame_count = 0
    detect_model = detect.Detect()

    while True:
        # grab next frame
        # frame = stream.read()
        frame = video.read()

        key = cv2.waitKey(1) & 0xFF

        if frame_count % 20 != 0:
            frame_count += 1
            continue

        if frame_count % 10 != 0:
            frame_count += 1
            continue
        frame_count += 1

        frame = detect_model.predect_with_reid(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':
    detect_process()