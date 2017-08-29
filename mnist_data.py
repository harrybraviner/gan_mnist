#! /usr/bin/python3

import numpy as np

class ImageSet:
    def __init__(self, filename):
        self.f = open(filename, "rb")

        self.f.seek(4)
        self.N_images = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(8)
        self.rows = int.from_bytes(self.f.read(4), byteorder="big")

        self.f.seek(12)
        self.cols = int.from_bytes(self.f.read(4), byteorder="big")

        self.next_index = 1

    def getImageAsBytes(self, index):
        if (index < 1 or index > self.N_images):
            return None
        else:
            self.f.seek(16 + (index-1)*self.rows*self.cols)
            return self.f.read(self.rows*self.cols)

    def getImageAsFloatArray(self, index):
        return np.reshape([float(x)/255.0 for x in self.getImageAsBytes(index)],
                          (self.rows, self.cols))

class LabelSet:
    def __init__(self, filename):
        self.f = open(filename, "rb")
        self.f.seek(4)
        self.N_labels = int.from_bytes(self.f.read(4), byteorder="big")

    def getLabel(self, index):
        if (index < 1 or index > self.N_labels):
            return None
        else:
            self.f.seek(8 + (index-1))
            return int.from_bytes(self.f.read(1), byteorder="big")

    def getOneHotLabel(self, index):
        label = self.getLabel(index)
        one_hot = np.zeros(10)
        one_hot[label] = 1
        return one_hot

class ImageAndLabelSet:
    def __init__(self, filename_images, filename_labels, training_fraction, total_fraction):
        self.image_set = ImageSet(filename_images)
        self.label_set = LabelSet(filename_labels)

        self.N_images = int(total_fraction * self.image_set.N_images)

        self.N_train = int(training_fraction * self.N_images)
        self.N_validation = self.N_images - self.N_train
        self.next_training_index = 1
        self.next_validation_index = self.N_train + 1

    def _getNextBatch(self, min_index, max_index, start_index, batchSize):
        next_index = start_index
        image_size = self.image_set.rows * self.image_set.cols
        image_batch = np.reshape([], (0, self.image_set.rows, self.image_set.cols, 1))
        label_batch = np.reshape([], (0, 10))
        remaining_images = batchSize
        while (remaining_images > 0):
            image_batch = np.concatenate([image_batch, np.reshape(self.image_set.getImageAsFloatArray(next_index), (1, self.image_set.rows, self.image_set.cols, 1))], axis=0)
            label_batch = np.concatenate([label_batch, np.reshape(self.label_set.getOneHotLabel(next_index), (1, 10))], axis=0)
            remaining_images -= 1
            next_index += 1
            if (next_index > max_index):
                next_index = min_index
        return next_index, image_batch, label_batch

    def getNextTrainingBatch(self, batch_size):
        self.next_training_index, image_batch, label_batch =\
            self._getNextBatch(1, self.N_train, self.next_training_index, batch_size)
        return image_batch, label_batch

    def getNextValidationBatch(self, batch_size):
        self.next_validation_index, image_batch, label_batch =\
            self._getNextBatch(self.N_train + 1, self.N_images, self.next_validation_index, batch_size)
        return image_batch, label_batch

    def getEntireTrainingSetAsBatches(self, batch_size):
        next_index = 1
        done = False
        while (not done):
            if (self.N_train - next_index + 1 > batch_size):
                next_index, image_batch, label_batch = self._getNextBatch(1, self.N_train, next_index, batch_size)
                yield batch_size, image_batch, label_batch
            else:
                next_index, image_batch, label_batch = self._getNextBatch(1, self.N_train, next_index, (self.N_train - next_index + 1))
                done = True
                yield batch_size, image_batch, label_batch

    def getEntireValidationSetAsBatches(self, batch_size):
        next_index = self.N_train + 1
        done = False
        while (not done):
            if (self.N_images - next_index + 1 > batch_size):
                next_index, image_batch, label_batch = self._getNextBatch(self.N_train + 1, self.N_images, next_index, batch_size)
                yield batch_size, image_batch, label_batch
            else:
                next_index, image_batch, label_batch = self._getNextBatch(self.N_train + 1, self.N_images, next_index, (self.N_images - next_index + 1))
                done = True
                yield batch_size, image_batch, label_batch

def getTrainingSet(fraction_to_use = 1.0):
    return ImageAndLabelSet("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", 0.8, fraction_to_use)

def getTestSet(fraction_to_use = 1.0):
    return ImageAndLabelSet("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", 1.0, fraction_to_use)
