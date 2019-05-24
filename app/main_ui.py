# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(886, 670)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.local_ratio = QtWidgets.QRadioButton(self.centralWidget)
        self.local_ratio.setObjectName("local_ratio")
        self.gridLayout.addWidget(self.local_ratio, 2, 0, 1, 1)
        self.camera_ratio = QtWidgets.QRadioButton(self.centralWidget)
        self.camera_ratio.setChecked(True)
        self.camera_ratio.setObjectName("camera_ratio")
        self.gridLayout.addWidget(self.camera_ratio, 1, 0, 1, 1)
        self.video_label = QtWidgets.QLabel(self.centralWidget)
        self.video_label.setFrameShape(QtWidgets.QFrame.Box)
        self.video_label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.video_label.setText("")
        self.video_label.setObjectName("video_label")
        self.gridLayout.addWidget(self.video_label, 0, 0, 1, 4)
        self.detection_button = QtWidgets.QPushButton(self.centralWidget)
        self.detection_button.setObjectName("detection_button")
        self.gridLayout.addWidget(self.detection_button, 1, 3, 1, 1)
        self.reid_video_button = QtWidgets.QPushButton(self.centralWidget)
        self.reid_video_button.setObjectName("reid_video_button")
        self.gridLayout.addWidget(self.reid_video_button, 2, 3, 1, 1)
        self.open_video_button = QtWidgets.QPushButton(self.centralWidget)
        self.open_video_button.setObjectName("open_video_button")
        self.gridLayout.addWidget(self.open_video_button, 1, 2, 1, 1)
        self.get_query_img_button = QtWidgets.QPushButton(self.centralWidget)
        self.get_query_img_button.setObjectName("get_query_img_button")
        self.gridLayout.addWidget(self.get_query_img_button, 2, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 886, 22))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.local_ratio.setText(_translate("MainWindow", "local"))
        self.camera_ratio.setText(_translate("MainWindow", "camera"))
        self.detection_button.setText(_translate("MainWindow", "行人检测"))
        self.reid_video_button.setText(_translate("MainWindow", "行人查询(再识别)"))
        self.open_video_button.setText(_translate("MainWindow", "打开视频"))
        self.get_query_img_button.setText(_translate("MainWindow", "关闭摄像头"))


