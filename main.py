from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ui import Ui_MainWindow

import numpy as np
from random import randrange
from keras.models import load_model
import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *
import h5py
import numpy as np


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.gui = Ui_MainWindow()
        self.gui.setupUi(self)

        self.gui.loadimage.clicked.connect(self.up_img)
        self.gui.fiveone.clicked.connect(self.Show_Train_Images)
        self.gui.fivetwo.clicked.connect(self.Show_Distribution)
        self.gui.fivethree.clicked.connect(self.Show_Model_Structure)
        self.gui.fivefour.clicked.connect(self.Show_Comparison)
        self.gui.fivefive.clicked.connect(self.Inference)

    def up_img(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.image = plt.imread(filename, format=None)

    def Show_Train_Images(self):

        dataset = tf.keras.utils.image_dataset_from_directory("inference_dataset/", color_mode='rgb', image_size=(224, 224), shuffle=False)
        for x, y in dataset:
            pass
        index1 = randrange(0, 5)
        index2 = randrange(5, 10)
        plt.figure("cat & dog")
        plt.subplot(1, 2, 1)
        plt.title('cat')
        plt.imshow(np.array(x[index1], dtype=np.uint8))
        plt.subplot(1, 2, 2)
        plt.title('dog')
        plt.imshow(np.array(x[index2], dtype=np.uint8))
        plt.show()

    def Show_Distribution(self):
        img = cv2.imread('train_dataset.jpg')
        cv2.imshow('class_distribution', img)
        cv2.waitKey(0)

    def Show_Model_Structure(self):
        model = load_model('model.h5')
        model.summary()

    def Show_Comparison(self):
        img = cv2.imread('loss compare.jpg')
        cv2.imshow('Accuracy Comparison', img)
        cv2.waitKey(0)


    def Inference(self):
        import tensorflow as tf
        from keras.models import load_model
        import numpy as np
        import skimage
        from skimage.transform import resize
        import matplotlib.pyplot as plt

        model = load_model('model.h5')

        img = self.image
        new_img = tf.image.resize(img, [224, 224])
        x = np.array(new_img)
        x = np.reshape(x, [-1, 224, 224, 3])/255.

        result = model.predict(x)
        ax = plt.subplot(1, 1, 1)
        plt.imshow(img)
        print(result)
        if result > 0.5:
            ax.set_title("Prediction = dog")
        else:
            ax.set_title("Prediction = cat")
        plt.show()


if __name__ == '__main__':
       import sys
       app = QtWidgets.QApplication(sys.argv)
       mainWindow = MainWindow_controller()
       mainWindow.show()
       sys.exit(app.exec_())