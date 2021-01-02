#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#



#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    cwd = os.getcwd()
    INPUT_PATH = cwd + r'\MNIST'
    TRAINING_IMAGES_FILEPATH = join(INPUT_PATH, r'files/train-images.idx3-ubyte')
    TRAINING_LABELS_FILEPATH = join(INPUT_PATH, r'files/train-labels.idx1-ubyte')
    TEST_IMAGES_FILEPATH = join(INPUT_PATH, r'files/t10k-images.idx3-ubyte')
    TEST_LABELS_FILEPATH = join(INPUT_PATH, r'files/t10k-labels.idx1-ubyte')


    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        img_train, lable_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        img_test, lable_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (img_train, lable_train), (img_test, lable_test)

    #


# Verify Reading Dataset via MnistDataloader class
#


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15);
        index += 1
    plt.show()

def test_main():
    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(MnistDataloader.TRAINING_IMAGES_FILEPATH, MnistDataloader.TRAINING_LABELS_FILEPATH,
                                       MnistDataloader.TEST_IMAGES_FILEPATH, MnistDataloader.TEST_LABELS_FILEPATH)
    (img_train, lable_train), (img_test, lable_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for r in range(0,20):
        #r = random.randint(1, 60000)
        images_2_show.append(img_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(lable_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(img_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(lable_test[r]))

    show_images(images_2_show, titles_2_show)


if __name__ == "__main__":
   test_main()