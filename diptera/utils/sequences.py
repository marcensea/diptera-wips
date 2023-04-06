
import os
import numpy as np
from numpy.random import default_rng
import cv2
from pathlib import Path

from tensorflow.keras.utils import to_categorical, Sequence


rng = default_rng()

def import_images_list(im_size, data_list, database, bounds, modifiers=None):

    """

    We use this function to import all the images of a path

    :param im_size: We define the size of the image we want to import
    :param data_list: The paths of the images
    :param index:
    :param modifiers: tab of tuple containing the type and the probability of each type if data augmentation
    :return: The list of all the images between the two indexes

    """

    train_list = []
    for k in range(bounds[0], bounds[1]):
        j = database[k]
        img_path = os.path.join(data_list[j])
        try:
            img = cv2.imread(img_path)
        except IOError as e:
            raise e("Path does not exist: %s" % img_path)
        x, y, channel = img.shape
        if im_size is not None:
            if im_size[0] > x | im_size[1] > y:
                im_size = (x, y)
        if im_size is not None:
            img = cv2.resize(img, im_size)
        if modifiers is not None:
            img = process(img, modifiers)
        train_list.append(img)
    return train_list



class DatalistGenerator(Sequence):
    """
    Generates the data for the training.

    We use this generator if we have a big dataset, it permits to avoid the save of all the images directly in the
    RAM of the GC, by generating the images batch per batch.

    :param im_size: a tuple representing the size of our image
    :param data_list: a list of all the path to the images we want to use.
    :param n_classes: the number of class we are using
    :param random_seed: boolean, set it to True if you want to randomize the data at the beginning of every epoch
    :param batch_size: a int representing the size of our batch
    :param y_true: a list representing the ground truth : must be same length than the number of images, you must
    name correctly your files so that the ground truth corresponds to the right image (instead of 1.jpg, 2.jpg, ...,
    10.jpg, you should use 01.jpg, 02.jpg, ..., 10.jpg). If you use different paths, just enter the ground_truth in the
    same list : the images of the 1st path will be imported first (the 1st path in the list data_path), then the 2nd ...
    So make sure to name the files correctly and to enter the paths in data_path in the right order.
    :param data_augmentation: a list of the objects for the data_augmentation. To see all the available objects, check
    in src_tf2/database/data_augmentation.py. You must first initialize your object, and then add it to the parameter
    data_augmentation.
    """

    def __init__(self, im_size, data_list, y_true, n_classes, randomize=True, batch_size=16,
                 data_augmentation=None, normalize=True):
        self.im_size = im_size
        self.data_list = data_list  # a tab of all the different paths
        self.batch_size = batch_size
        self.randomize = randomize
        self.normalize = normalize

        self.names = np.array([Path(filepath).name for filepath in data_list], dtype=str)  # the filename of samples
        self.n_classes = n_classes  # the number of different classes
        self.n_samples = len(data_list)  # the full number of samples
        self.index = np.arange(self.n_samples)  # an array containing 0 to samples
        self.data_augmentation = data_augmentation

        self.y_true = np.array(y_true)
        if self.randomize:
            perm = rng.permutation(self.n_samples)
            self.index, self.y_true = self.index[perm], self.y_true[perm]

    def __len__(self):
        """ Return the number of batches. """
        if np.floor(float(self.n_samples) / float(self.batch_size)) == float(self.n_samples) / float(self.batch_size):
            return int(np.floor(float(self.n_samples) / float(self.batch_size)))
        else:
            return int(np.floor(float(self.n_samples) / float(self.batch_size))) + 1

    def __getitem__(self, idx):
        """
        Return a batch of data.
        :param idx: the batch index
        :return: Tuple(image array, label array)
        """
        if self.batch_size * (idx + 1) > self.n_samples:
            idx_final = self.n_samples
        else:
            idx_final = self.batch_size * (idx + 1)
        x_database = import_images_list(self.im_size, self.data_list, self.index, [idx * self.batch_size, idx_final],
                                        modifiers=self.data_augmentation)
        if self.normalize:
            x_database = np.array(x_database, dtype=np.float32)
            x_database *= 1./255
        else:
            x_database = np.array(x_database)

        y_database = self.y_true[idx * self.batch_size: idx_final]
        if self.n_classes > 1:
            y_database = to_categorical(y_database, num_classes=self.n_classes)
        else:
            y_database = np.array(y_database)
        return x_database, y_database

    def on_epoch_end(self):
        if self.randomize:
            [self.index, self.y_true] = shuffle(self.index, self.y_true)  # method which shuffles the database and the
            # truth

    def get(self, i):
        """
        Return a single sample of data.
        :param i: a sample indice
        :return: Tuple(image, label)
        """
        sample = self.index[i]
        x = import_image(self.im_size, self.data_list[sample], modifiers=self.data_augmentation)
        if self.normalize:
            x = x.astype('float32') * 1./255
        y = self.y_true[i]
        if self.n_classes > 1:
            y = to_categorical(y, num_classes=self.n_classes)
        return x, y

    def reindex(self, index):
        """ Change the index of samples. Warning: Number of indices can be lower than n_samples. """
        self.index = index
        self.n_samples = len(self.index)
        self.names = self.names[index]
        self.y_true = self.y_true[index]


class HDF5_Generator(Sequence):
    """
    Generates the data for the training.

    :param im_size: a tuple representing the size of our image
    :param data_list: a list of all the path to the images we want to use.
    :param n_classes: the number of class we are using
    :param random_seed: boolean, set it to True if you want to randomize the data at the beginning of every epoch
    :param batch_size: a int representing the size of our batch
    """

    def __init__(self, data, labels, n_classes, randomize=True, batch_size=16,
                 data_augmentation=None, normalize=True):

        self.data = data
        self.labels = labels

        self.randomize = randomize
        self.normalize = normalize

        self.batch_size = batch_size
        self.n_classes = n_classes  # the number of different classes
        self.n_samples = len(data)  # the full number of samples
        self.index = np.arange(self.n_samples)  # an array containing 0 to samples
        self.data_augmentation = data_augmentation

        self.labels = np.array(labels)
        if self.randomize:
            perm = rng.permutation(self.n_samples)
            self.index, self.labels = self.index[perm], self.labels[perm]

    def __len__(self):
        """ Return the number of batches. """
        if np.floor(float(self.n_samples) / float(self.batch_size)) == float(self.n_samples) / float(self.batch_size):
            return int(np.floor(float(self.n_samples) / float(self.batch_size)))
        else:
            return int(np.floor(float(self.n_samples) / float(self.batch_size))) + 1

    def __getitem__(self, idx):
        """
        Return a batch of data.
        :param idx: the batch index
        :return: Tuple(image array, label array)
        """
        if self.batch_size * (idx + 1) > self.n_samples:
            idx_final = self.n_samples
        else:
            idx_final = self.batch_size * (idx + 1)
        x_database = import_images_list(self.im_size, self.data_list, self.index, [idx * self.batch_size, idx_final],
                                        modifiers=self.data_augmentation)
        if self.normalize:
            x_database = np.array(x_database, dtype=np.float32)
            x_database *= 1./255
        else:
            x_database = np.array(x_database)

        y_database = self.y_true[idx * self.batch_size: idx_final]
        if self.n_classes > 1:
            y_database = to_categorical(y_database, num_classes=self.n_classes)
        else:
            y_database = np.array(y_database)
        return x_database, y_database

    def on_epoch_end(self):
        if self.randomize:
            [self.index, self.y_true] = shuffle(self.index, self.y_true)  # method which shuffles the database and the
            # truth

    def get(self, i):
        """
        Return a single sample of data.
        :param i: a sample indice
        :return: Tuple(image, label)
        """
        sample = self.index[i]
        x = import_image(self.im_size, self.data_list[sample], modifiers=self.data_augmentation)
        if self.normalize:
            x = x.astype('float32') * 1./255
        y = self.y_true[i]
        if self.n_classes > 1:
            y = to_categorical(y, num_classes=self.n_classes)
        return x, y

    def reindex(self, index):
        """ Change the index of samples. Warning: Number of indices can be lower than n_samples. """
        self.index = index
        self.n_samples = len(self.index)
        self.names = self.names[index]
        self.y_true = self.y_true[index]

