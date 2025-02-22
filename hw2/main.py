from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import utils
import numpy as np
from ui import Ui_MainWindow
import cv2
import os
import matplotlib.pyplot as plt

class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.test = Ui_MainWindow()
        self.test.setupUi(self)

        self.test.LF.clicked.connect(self.LF)
        self.test.LIL.clicked.connect(self.LIL)
        self.test.LIR.clicked.connect(self.LIR)

        self.test.oneone.clicked.connect(self.draw_contour)
        self.test.onetwo.clicked.connect(self.count_rings)

        self.test.twoone.clicked.connect(self.corner_detection)
        self.test.twotwo.clicked.connect(self.Intrinsic_Matrix)
        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.test.comboBox.addItems(choices)
        self.test.comboBox.currentIndexChanged.connect(self.combo_box)
        self.test.twothree.clicked.connect(self.Extrinsic_Matrix)
        self.test.twofour.clicked.connect(self.Distortion_Matrix)
        self.test.twofive.clicked.connect(self.undistorted)

        self.test.threeone.clicked.connect(self.On_Board)
        self.test.threetwo.clicked.connect(self.Vertically)
        self.char_in_board = [
            [7, 5, 0],  # slot 1
            [4, 5, 0],  # slot 2
            [1, 5, 0],  # slot 3
            [7, 2, 0],  # slot 4
            [4, 2, 0],  # slot 5
            [1, 2, 0]  # slot 6
        ]
        self.q4_1 = False
        self.test.fourone.clicked.connect(self.Stereo)
        self.test.fourtwo.clicked.connect(self.checkDisparity)

    def LF(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")

    def LIL(self):
        self.filename_l, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.img1 = cv2.imread(self.filename_l)

    def LIR(self):
        self.filename_r, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.img2 = cv2.imread(self.filename_r)

    def draw_contour(self):
        img1 = self.img1
        img2 = self.img2

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        ret1, thresh1 = cv2.threshold(blur1, 127, 255, 0)
        ret2, thresh2 = cv2.threshold(blur2, 127, 255, 0)

        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt1 in contours1:
            cv2.drawContours(img1, contours1, -1, (255, 200, 0), 2)
        for cnt2 in contours2:
            cv2.drawContours(img2, contours2, -1, (255, 200, 0), 2)

        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)

    def count_rings(self):
        img1 = self.img1
        img2 = self.img2
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        ret1, thresh1 = cv2.threshold(blur1, 127, 255, 0)
        ret2, thresh2 = cv2.threshold(blur2, 127, 255, 0)
        contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ans1 = 0
        ans2 = 0
        for cnt1 in contours1:
            area = cv2.contourArea(cnt1)
            if area > 500:
                ans1 = ans1 + 1
        for cnt2 in contours2:
            area = cv2.contourArea(cnt2)
            if area > 500:
                ans2 = ans2 + 1
        string1 = "There are "+str(ans1 / 2)+" rings in img_1.jpg"
        string2 = "There are "+str(ans2 / 2)+" rings in img_1.jpg"

        self.test.label.setText(string1)
        self.test.label_2.setText(string2)

    def corner_detection(self):
        w = 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename

            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1/3, fy=1/3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(res, (w, h), corners, ret)
                cv2.imshow('findCorners', res)
                cv2.waitKey(500)
        cv2.destroyWindow('findCorners')

    def Intrinsic_Matrix(self):

        w = 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []

        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename

            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:

                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        print("Intrinsic matrix")
        print(mtx)

    def combo_box(self):
        self.num = self.test.comboBox.currentText()

    def Extrinsic_Matrix(self):
        w = 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []

        path = self.folder_path + "/" + '%s' % self.num + '.bmp'

        img = cv2.imread(path)
        res = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        cv2.imshow("img", gray)

        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        R = cv2.Rodrigues(rvecs[0])
        T = np.zeros((3, 1), np.float32)

        T[0][0] = tvecs[0][0]
        T[1][0] = tvecs[0][1]
        T[2][0] = tvecs[0][2]

        Extrinsic = np.hstack((R[0], T))
        print("Extrinsic_Matrix")
        print(Extrinsic)

    def Distortion_Matrix(self):
        w = 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []

        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename

            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        print("Distortion matrix")
        print(dist)

    def undistorted(self):
        w = 11
        h = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []

        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename

            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

            if ret == True:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename

            img2 = cv2.imread(path)
            res2 = cv2.resize(img2, (0, 0), fx=1 / 3, fy=1 / 3)
            h, w = res2.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
            dst = cv2.undistort(res2, mtx, dist, None, newcameramtx)
            cv2.imshow('undistored', dst)
            cv2.moveWindow('undistored', 683, 0)
            cv2.imshow('distored', res2)

            cv2.waitKey(500)

    def On_Board(self):
        w = 11
        h = 8
        string = self.test.text.text()[:6]
        fs = cv2.FileStorage("alphabet_lib_onboard.txt", cv2.FILE_STORAGE_READ)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((h * w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []
        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename
            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1/3, fy=1/3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corner = cv2.findChessboardCorners(gray, (w, h), None)
            if ret == True:
                obj_points.append(objp)
                img_points.append(corner)
        images = []
        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename
            image = cv2.imread(path)
            res = cv2.resize(image, (0, 0), fx=1/3, fy=1/3)
            images.append(res)

        for index, image in enumerate(images):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3, 1)

                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []

                    for each_line in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        each_line = np.add(each_line, ach)
                        image_points, jac = cv2.projectPoints(each_line, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)

                    draw_image = draw_image.copy()
                    for line in line_list:
                        line = line.reshape(2, 2)
                        draw_image = cv2.line(draw_image, (line[0].astype(int)), (line[1].astype(int)), (0, 0, 255), 5, cv2.LINE_AA)

            cv2.imshow("WORD ON BOARD", draw_image)
            cv2.waitKey(1000)

    def Vertically(self):
        w = 11
        h = 8
        string = self.test.text.text()[:6]
        fs = cv2.FileStorage("alphabet_lib_vertical.txt", cv2.FILE_STORAGE_READ)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((h * w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        obj_points = []
        img_points = []
        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename
            img = cv2.imread(path)
            res = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, corner = cv2.findChessboardCorners(gray, (w, h), None)
            if ret == True:
                obj_points.append(objp)
                img_points.append(corner)
        images = []
        for filename in os.listdir(self.folder_path):
            path = self.folder_path + "/" + filename
            image = cv2.imread(path)
            res = cv2.resize(image, (0, 0), fx=1 / 3, fy=1 / 3)
            images.append(res)

        for index, image in enumerate(images):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3, 1)

                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []

                    for each_line in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        each_line = np.add(each_line, ach)
                        image_points, jac = cv2.projectPoints(each_line, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)

                    draw_image = draw_image.copy()
                    for line in line_list:
                        line = line.reshape(2, 2)
                        draw_image = cv2.line(draw_image, (line[0].astype(int)), (line[1].astype(int)), (0, 0, 255), 5,
                                              cv2.LINE_AA)

            cv2.imshow("Vertically", draw_image)
            cv2.waitKey(1000)

    def Stereo(self):
        img_l = cv2.imread(self.filename_l, 0)
        img_r = cv2.imread(self.filename_r, 0)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(img_l, img_r)/16

        cv2.namedWindow("imL", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('imL', img_l)
        cv2.namedWindow("imR", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('imR', img_r)
        plt.imshow(disparity, 'gray')
        plt.show()
        cv2.waitKey(0)
        self.q4_1 = True

    def checkDisparity(self):
        if not self.q4_1:
            self.Stereo()
        cv2.namedWindow("Checking Disparity", cv2.WINDOW_GUI_EXPANDED)

        img_l = cv2.imread(self.filename_l)
        img_r = cv2.imread(self.filename_r)

        hL, wL = img_l.shape[:2]
        merge_image = cv2.hconcat([img_l, img_r])

        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        def onmouse(event, x, y, flags, param):
            nonlocal merge_image
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cur_img = merge_image.copy()
                cur_x, cur_y = x, y
                if cur_x < 0:
                    cur_x = 0
                elif cur_x >= wL:
                    cur_x = wL - 1
                if cur_y < 0:
                    cur_y = 0
                elif cur_y >= hL:
                    cur_y = hL - 1

                matcher = cv2.StereoBM_create(256, 25)
                delta_pos = matcher.compute(img_l, img_r)[cur_y, cur_x]//16

                print("disparity value at ({},{}): {}".format(cur_y, cur_x, delta_pos))
                if delta_pos > 0:
                    x_right = cur_x - delta_pos + wL
                    cur_image = cv2.circle(cur_img, (x_right, cur_y), radius=20, color=(0, 255, 0), thickness=-1)
                    cv2.imshow("Checking Disparity", cur_image)

        cv2.imshow("Checking Disparity", merge_image)
        cv2.setMouseCallback("Checking Disparity", onmouse)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow_controller()
    mainWindow.show()
    sys.exit(app.exec_())
