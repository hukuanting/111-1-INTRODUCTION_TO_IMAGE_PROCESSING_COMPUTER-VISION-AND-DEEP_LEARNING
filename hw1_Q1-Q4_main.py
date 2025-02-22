from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
from gui import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.gui = Ui_MainWindow()
        self.gui.setupUi(self)

        self.gui.loadimage.clicked.connect(self.up_img1)
        self.gui.threeone.clicked.connect(self.Gaussian_blur)
        self.gui.threetwo.clicked.connect(self.Sobel_X)
        self.gui.threethree.clicked.connect(self.Sobel_Y)
        self.gui.threefour.clicked.connect(self.Magnitude)
        self.gui.fourone.clicked.connect(self.Resize)
        self.gui.fourtwo.clicked.connect(self.Translation)
        self.gui.fourthree.clicked.connect(self.Rotation_Scaling)
        self.gui.fourfour.clicked.connect(self.Shearing)

    def up_img1(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.image = cv2.imread(filename)

    def Gaussian_blur(self):

        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def convolution(kernel, image):
            n, m = image.shape
            new_img = []
            for i in range(n-3):
                line = []
                for j in range(m-3):
                    a = image[i:i+3, j:j+3]
                    line.append(np.sum(np.multiply(kernel, a)))
                new_img.append(line)
            return np.array(new_img)

        y, x = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        img_new = convolution(gaussian_kernel, gray)

        row1, col1 = img_new.shape
        for row in range(row1):
            for col in range(col1):
                img_new[row][col] = abs(img_new[row][col])
        blurred = np.array(img_new, 'uint8')
        cv2.imwrite('blurred.jpg', blurred)
        cv2.imshow('Gaussian_Blur', blurred)

    def Sobel_X(self):
        def convolution(kernel, image):
            n, m = image.shape
            new_img = []
            for i in range(n-3):
                line = []
                for j in range(m-3):
                    a = image[i:i+3, j:j+3]
                    line.append(np.sum(np.multiply(kernel, a)))
                new_img.append(line)
            return np.array(new_img)

        image = cv2.imread('blurred.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        Sobel_X_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        Sobel_X = convolution(Sobel_X_kernel, gray)
        row1, col1 = Sobel_X.shape
        for row in range(row1):
            for col in range(col1):
                Sobel_X[row][col] = abs(Sobel_X[row][col])

        Sobel_X = np.array(Sobel_X, 'uint8')

        cv2.imwrite('Sobel_X.jpg', Sobel_X)
        cv2.imshow("Sobel_X",Sobel_X)

    def Sobel_Y(self):

        image = cv2.imread("blurred.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Sobel_Y_kernel = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        def convolution(kernel, image):
            n, m = image.shape
            new_img = []
            for i in range(n - 3):
                line = []
                for j in range(m - 3):
                    a = image[i:i + 3, j:j + 3]
                    line.append(np.sum(np.multiply(kernel, a)))
                new_img.append(line)
            return np.array(new_img)

        Sobel_Y = convolution(Sobel_Y_kernel, gray)

        row1, col1 = Sobel_Y.shape
        for row in range(row1):
            for col in range(col1):
                Sobel_Y[row][col] = abs(Sobel_Y[row][col])

        Sobel_Y = np.array(Sobel_Y, 'uint8')

        cv2.imwrite('Sobel_Y.jpg', Sobel_Y)
        cv2.imshow("Sobel_Y", Sobel_Y)

    def Magnitude(self):
        image1 = cv2.imread('Sobel_X.jpg')
        image2 = cv2.imread('Sobel_Y.jpg')
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        magnitude = np.hypot(gray1, gray2)
        row1, col1 = magnitude.shape
        for row in range(row1):
            for col in range(col1):
                magnitude[row][col] = abs(magnitude[row][col])
        magnitude = np.array(magnitude, 'uint8')
        cv2.imshow('Magnitude', magnitude)

    def Resize(self):
        img = self.image
        H = np.float32([[1, 0, 0], [0, 1, 0]])
        img_resized = cv2.resize(img, (215, 215))
        res = cv2.warpAffine(img_resized, H, (430, 430))  # 需要图像、变换矩阵、变换后的大小
        cv2.imshow("res", res)
        cv2.waitKey(0)

    def Translation(self):
        img = self.image
        H1 = np.float32([[1, 0, 0], [0, 1, 0]])
        H2 = np.float32([[1, 0, 215], [0, 1, 215]])

        img_resized = cv2.resize(img, (215, 215))

        res1 = cv2.warpAffine(img_resized, H1, (430, 430))
        res2 = cv2.warpAffine(img_resized, H2, (430, 430))

        res = cv2.addWeighted(res1, 1, res2, 1, 0)

        cv2.imshow("res", res)
        cv2.waitKey(0)

    def Rotation_Scaling(self):
        img = self.image
        res = cv2.resize(img, (108, 108))

        H1 = np.float32([[1, 0, 107], [0, 1, 161]])
        H2 = np.float32([[1, 0, 215], [0, 1, 161]])
        M = cv2.getRotationMatrix2D((54, 54), 45, 1 / 2 ** 0.5)

        rot = cv2.warpAffine(res, M, (430, 430))
        scal1 = cv2.warpAffine(rot, H1, (430, 430))
        scal2 = cv2.warpAffine(rot, H2, (430, 430))

        rot_scal = cv2.addWeighted(scal1, 1, scal2, 1, 0)

        cv2.imshow("rot_scal", rot_scal)
        cv2.waitKey(0)

    def Shearing(self):
        img = self.image
        res = cv2.resize(img, (108, 108))

        H1 = np.float32([[1, 0, 107], [0, 1, 161]])
        H2 = np.float32([[1, 0, 215], [0, 1, 161]])
        M = cv2.getRotationMatrix2D((54, 54), 45, 1 / 2 ** 0.5)

        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [100, 50], [100, 250]])
        P = cv2.getAffineTransform(pts1, pts2)

        rot = cv2.warpAffine(res, M, (430, 430))
        scal1 = cv2.warpAffine(rot, H1, (430, 430))
        scal2 = cv2.warpAffine(rot, H2, (430, 430))

        shear1 = cv2.warpAffine(scal1, P, (430, 430))
        shear2 = cv2.warpAffine(scal2, P, (430, 430))

        rot_scal_shear = cv2.addWeighted(shear1, 1, shear2, 1, 0)

        cv2.imshow("rot_scal_shear", rot_scal_shear)
        cv2.waitKey(0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow_controller()
    mainWindow.show()
    sys.exit(app.exec_())
