from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ui import Ui_MainWindow

from random import randrange
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input

import keras
import tensorflow as tf
from keras.datasets import cifar10
from keras.applications import VGG19
from keras.models import load_model

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.gui = Ui_MainWindow()
        self.gui.setupUi(self)

        self.gui.loadimage.clicked.connect(self.up_img)
        self.gui.one.clicked.connect(self.Show_Train_Images)
        self.gui.two.clicked.connect(self.Show_Model_Structure)
        self.gui.three.clicked.connect(self.Show_Data_Augmentation)
        self.gui.four.clicked.connect(self.Show_Accuracy_and_Loss)
        self.gui.five.clicked.connect(self.Inference)

    def up_img(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.image = plt.imread(filename, format=None)

    def Show_Train_Images(self):

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plt.figure(figsize=(6, 6), facecolor='w')
        for i in range(3):
            for j in range(3):
                index = randrange(0, 50000)
                plt.subplot(3, 3, i * 3 + j + 1)
                plt.title("{}".format(text[y_train[index][0]]), fontproperties="normal")
                plt.imshow(x_train[index])
                plt.axis('off')

        plt.show()

    def Show_Model_Structure(self):

        vgg19_model = VGG19(include_top=True, weights='imagenet')
        model = Sequential()

        for layer in vgg19_model.layers[:-1]:
            model.add(layer)

        model.add(Dense(1024, activation='ReLU'))
        model.add(Dense(512, activation='ReLU'))
        model.add(Dense(10, activation='softmax'))
        model.summary()

    def Show_Data_Augmentation(self):
        image = self.image
        def augment1(image):
            x = tf.keras.preprocessing.image.random_rotation(
                image,
                30,
                row_axis=1,
                col_axis=2,
                channel_axis=0,
                fill_mode='nearest',
                cval=0.0,
                interpolation_order=1
            )
            x = tf.image.random_crop(x, size=[224, 224, 3])
            x = tf.image.resize(x, [224, 224])
            x = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

            return x

        def augment2(image: tf.Tensor) -> tf.Tensor:
            """Flip augmentation

            Args:
                image: Image to flip

            Returns:
                Augmented image
            """
            y = tf.image.random_flip_left_right(image)
            y = tf.image.random_flip_up_down(y)
            y = tf.cast(y, tf.float32)
            y = tf.image.resize(y, [224, 224])
            y = (y / 255.0)
            y = tf.image.random_crop(y, size=[180, 224, 3])
            y = tf.image.random_brightness(y, max_delta=0.5)

            return y

        def augment3(image: tf.Tensor) -> tf.Tensor:
            """Rotation augmentation

            Args:
                image: Image

            Returns:
                Augmented image
            """

            z = tf.image.random_hue(image, 0.1)
            z = tf.image.random_saturation(z, 0.8, 1.2)
            z = tf.image.random_contrast(z, 0.7, 1.3)
            z = tf.image.rot90(z, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            z = tf.image.random_crop(z, size=[180, 224, 3])
            return z

        new_img1 = augment1(image)
        new_img2 = augment2(image)
        new_img3 = augment3(image)

        fig = plt.figure()

        plt.subplot(1, 3, 1)
        plt.title('new_img1')
        plt.axis('off')
        plt.imshow(new_img1)
        plt.subplot(1, 3, 2)
        plt.title('new image2')
        plt.axis('off')
        plt.imshow(new_img2)
        plt.subplot(1, 3, 3)
        plt.title('new image3')
        plt.axis('off')
        plt.imshow(new_img3)
        plt.show()

    def Show_Accuracy_and_Loss(self):
        img = cv2.imread('plot.jpg')
        cv2.imshow('plot', img)
        cv2.waitKey(0)

    def Inference(self):

        import tensorflow as tf
        from keras.models import load_model
        import numpy as np
        from skimage.transform import resize
        import matplotlib.pyplot as plt

        model = load_model('model.h5')
        img = self.image
        x_batch = []

        CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        new_img = tf.image.resize(img, [224, 224])
        x_batch.append(new_img)
        x = np.array(x_batch)

        plt.imshow(img)

        plt.text(50, -5, "Prediction Label = " + CATEGORIES[np.argmax(model.predict(x))])

        print(model.predict(x))
        numbers = model.predict(x)
        numbers = np.reshape(numbers, [10])
        plt.text(50, -20, f"Confidence = {numbers[np.argmax(model.predict(x))]}")

        plt.show()


if __name__ == '__main__':
       import sys
       app = QtWidgets.QApplication(sys.argv)
       mainWindow = MainWindow_controller()
       mainWindow.show()
       sys.exit(app.exec_())
