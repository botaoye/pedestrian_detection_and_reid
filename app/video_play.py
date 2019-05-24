import cv2
import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from detection import detect
from reid import re_id


class Display:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd

        # 默认视频源为相机
        self.ui.camera_ratio.setChecked(True)
        self.isCamera = True

        # 信号槽设置
        ui.open_video_button.clicked.connect(lambda:self.Open("open_video"))
        ui.get_query_img_button.clicked.connect(self.Close)
        ui.camera_ratio.clicked.connect(self.radioButtonCam)
        ui.local_ratio.clicked.connect(self.radioButtonFile)

        # 检测按钮和再识别按钮
        ui.detection_button.clicked.connect(lambda:self.Open("detect"))
        ui.reid_video_button.clicked.connect(lambda:self.Open("reid"))

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def radioButtonCam(self):
        self.isCamera = True

    def radioButtonFile(self):
        self.isCamera = False

    def Open(self, method):
        if not self.isCamera:
            self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, 'Choose file', '', '*.mp4')
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            # 下面两种rtsp格式都是支持的
            # cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126/main/Channels/1")
            self.cap = cv2.VideoCapture(0)

        # 创建视频显示线程
        if method == "open_video":
            th = threading.Thread(target=self.Display)
            th.start()
        elif method == "detect":
            th = threading.Thread(target=self.detect)
            th.start()
        elif method == "reid":
            th = threading.Thread(target=self.reid)
            th.start()

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def Display(self):
        # self.ui.Open.setEnabled(False)
        # self.ui.Close.setEnabled(True)

        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.video_label.setPixmap(QPixmap.fromImage(img))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.video_label.clear()
                self.cap.release()
                # self.ui.Close.setEnabled(False)
                # self.ui.Open.setEnabled(True)
                break


    def detect(self):
        frame_count = 0
        detect_model = detect.Detect()
        last_frame = None

        while self.cap.isOpened():
            success, frame = self.cap.read()

            # 运行detect
            if frame_count % 10 != 0:
                frame_count += 1
                continue
            frame_count += 1

            frame = detect_model.predict(frame)

            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.video_label.setPixmap(QPixmap.fromImage(img))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.video_label.clear()
                self.cap.release()
                # self.ui.Close.setEnabled(False)
                # self.ui.Open.setEnabled(True)
                break

    def reid(self):
        frame_count = 0
        detect_model = detect.Detect()

        while self.cap.isOpened():
            success, frame = self.cap.read()

            # 运行detect
            if frame_count % 20 != 0:
                frame_count += 1
                continue
            frame_count += 1

            frame = detect_model.predect_with_reid(frame)
            # RGB转BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.video_label.setPixmap(QPixmap.fromImage(img))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.video_label.clear()
                self.cap.release()
                # self.ui.Close.setEnabled(False)
                # self.ui.Open.setEnabled(True)
                break