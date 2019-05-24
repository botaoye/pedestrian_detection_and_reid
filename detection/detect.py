from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
from reid import re_id

import time
from imutils.video import FPS, WebcamVideoStream, FileVideoStream

#    读取视频
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from detection.data import BaseTransform, VOC_CLASSES as labelmap
from detection.ssd import build_ssd



class Detect():
    def __init__(self):
        self.net = build_ssd('test', 300, 21)  # initialize SSD
        self.net.load_state_dict(torch.load("weights/detection_model.pth", map_location='cpu'))
        self.transform = BaseTransform(self.net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))
        self.re_id_model = re_id.Re_id()

    def predict(self, frame):

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        height, width = frame.shape[:2]
        print(height, width)
        x = torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = self.net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        print('detect success')
        return frame

    def predect_with_reid(self, frame):
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        height, width = frame.shape[:2]
        x = torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = self.net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

                detected_person = frame[int(pt[1]):int(pt[3]), int(pt[0]):int(pt[2])]
                cv2.imwrite("cache/detected_person.jpg", detected_person)
                time.sleep(0.1)
                score = self.re_id_model.compare('cache/query.jpg', 'cache/detected_person.jpg')
                print(score)

                if score > 0.7:
                    cv2.rectangle(frame,
                                  (int(pt[0]), int(pt[1])),
                                  (int(pt[2]), int(pt[3])),
                                  COLORS[i % 3], 2)
                    cv2.putText(frame, "Find!", (int(pt[0]), int(pt[1])),
                                FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1

        return frame