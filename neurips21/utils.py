import argparse
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import scipy.misc
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
from tabulate import tabulate


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_interval(eps_l=0.1, eps_r=1.0, num=10):
    num = int(num)
    assert num > 0 and (eps_l < eps_r)
    if num == 1:
        return [(eps_l + eps_r) / 2.0]
    elif num == 2:
        return [eps_l, eps_r]
    else:
        step = (eps_r - eps_l) / (num - 1)
        return [eps_l + step * i for i in range(num)]


class MNIST:
    def __init__(self, one_hot=True, shuffle=False, group_by_label=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot, group_by_label)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        if shuffle:
            self.shuffle_data()

    def load_data(self, one_hot, group_by_label):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (
            x_test,
            y_test,
        ) = mnist.load_data()
        ## x_train.shape = (60000, 28, 28), range = [0, 255]
        ## y_train.shape = (60000)

        x_train = np.reshape(x_train, [-1, 28, 28, 1])
        x_train = x_train.astype(np.float32) / 255
        x_test = np.reshape(x_test, [-1, 28, 28, 1])
        x_test = x_test.astype(np.float32) / 255

        if group_by_label:
            ind_train = np.argsort(y_train)
            ind_test = np.argsort(y_test)
            x_train, y_train = x_train[ind_train], y_train[ind_train]
            x_test, y_test = x_test[ind_test], y_test[ind_test]

        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]


class CIFAR10:
    def __init__(self, one_hot=True, shuffle=False, augument=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(one_hot)
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        if augument:
            self.x_train, self.y_train = self.augument_data()
        if shuffle:
            self.shuffle_data()

    def load_data(self, one_hot):
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        ##  x_train.shape = (50000, 32, 32, 3), range = [0, 255]
        ##  y_train.shape = (50000, 1)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        if one_hot:
            # convert to one-hot labels
            y_train = tf.keras.utils.to_categorical(y_train)
            y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def shuffle_data(self):
        ind = np.random.permutation(self.num_train)
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]

    def augument_data(self):
        image_generator = ImageDataGenerator(
            rotation_range=90,
            # zoom_range = 0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            # vertical_flip=True,
        )

        image_generator.fit(self.x_train)
        # get transformed images
        x_train, y_train = image_generator.flow(
            self.x_train, self.y_train, batch_size=self.num_train, shuffle=False
        ).next()

        return x_train, y_train


class Logger:
    def __init__(self, name="model", fmt=None, base="./logs"):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        if not os.path.exists(base):
            os.makedirs(base)
        self.path = os.path.join(base, name + "_" + str(time.time()))

        self.logs = self.path + ".csv"
        self.output = self.path + ".out"

        def prin(*args):
            str_to_write = " ".join(map(str, args))
            with open(self.output, "a") as f:
                f.write(str_to_write + "\n")
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ["%s"] + [self.fmt[name] if name in self.fmt else ".1f" for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ["epoch"] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ["epoch"] + names, tablefmt="plain", floatfmt=fmt).split("\n")[1])

    def save(self):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=["t", key]).set_index("t")
            else:
                df = DataFrame(self.scalar_metrics[key], columns=["t", key]).set_index("t")
                result = result.join(df, how="outer")
        result.to_csv(self.logs)

        self.print("The log/output have been saved to: " + self.path + " + .csv/.out")


class Experiment:
    pass


def identity(images):
    images = tf.convert_to_tensor(images)
    return tf.identity(images)


def fliplr(images, prob=1.0):
    images = tf.convert_to_tensor(images)
    do_flip = tf.random_uniform([]) < prob
    return tf.cond(do_flip, lambda: tf.image.flip_left_right(images), lambda: images)


def flipud(images):
    images = tf.convert_to_tensor(images)
    return tf.image.flip_up_down(images)


def adjust_brightness(images, delta=0.1, prob=1.0):
    images = tf.convert_to_tensor(images)
    do_adjust_brightness = tf.random_uniform([]) < prob
    return tf.cond(do_adjust_brightness, lambda: tf.image.adjust_brightness(images, delta=delta), lambda: images)


def adjust_gamma(images, gamma=1.3):
    images = tf.convert_to_tensor(images)
    return tf.image.adjust_gamma(images, gamma=gamma)


def adjust_contrast(images, contrast_factor=0.7):
    images = tf.convert_to_tensor(images)
    return tf.image.adjust_contrast(images, contrast_factor)


# support for tf 1.12.0 or higher (not support batch processing!!)
def adjust_jpeg_quality(images, quality=50):
    images = tf.convert_to_tensor(images)
    batch = images.shape[0]
    outputs = []
    for i in range(batch):
        outputs.append(tf.image.adjust_jpeg_quality(images[i], quality))
    return tf.stack(outputs, axis=0)


def crop_and_resize(images, boxes=[[0.1, 0.1, 0.9, 0.9]], box_ind=[0], crop_size=[32, 32]):
    images = tf.convert_to_tensor(images)
    batch = tf.shape(images)[0]
    boxes = tf.tile(boxes, [batch, 1])
    box_ind = tf.range(batch)

    return tf.image.crop_and_resize(images, boxes=boxes, box_ind=box_ind, crop_size=crop_size)


def rotate(images, angles):
    images = tf.convert_to_tensor(images)

    return tf.contrib.image.rotate(images, angles * np.pi / 180)


# rotation series
def rot30(images):
    return rotate(images, 30)


def rot60(images):
    return rotate(images, 60)


def rot90(images):
    return rotate(images, 90)


def rot120(images):
    return rotate(images, 120)


def rot150(images):
    return rotate(images, 150)


def rot180(images):
    return rotate(images, 180)


def rot30_(images):
    return rotate(images, -30)


def rot60_(images):
    return rotate(images, -60)


def rot90_(images):
    return rotate(images, -90)


def rot120_(images):
    return rotate(images, -120)


def rot150_(images):
    return rotate(images, -150)


def rotate_random(images, minval=-np.pi / 2, maxval=np.pi / 2):
    images = tf.convert_to_tensor(images)

    return tf.contrib.image.rotate(images, tf.random_uniform((), minval=minval, maxval=maxval))


def test_mnist():
    print("Testing MNIST dataloader...")
    data = MNIST()
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = MNIST(one_hot=False)
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print(data.y_train[0:10])
    data = MNIST(shuffle=True, one_hot=False)
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print(data.y_train[0:10])


def test_cifar10():
    print("Testing CIFAR10 dataloader...")
    data = CIFAR10()
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    data = CIFAR10(one_hot=False)
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print(data.y_train[0:10])
    data = CIFAR10(shuffle=True, one_hot=False)
    print(data.x_train.shape, data.y_train.shape, data.x_test.shape, data.y_test.shape)
    print(data.y_train[0:10])


def test_augmentation():
    print("Testing image augmentation ops...")
    data = CIFAR10()
    images = data.x_train[:5]
    # images = (images * 255).astype(np.uint8)

    # tensorflow image library
    with tf.Session() as sess:
        images_identity = identity(images)
        images_fliplr = fliplr(images)
        images_flipud = flipud(images)
        images_crop_and_resize = crop_and_resize(images)
        images_brightness = adjust_brightness(images)
        images_contrast = adjust_contrast(images)
        images_gamma = adjust_gamma(images)
        images_jpeg = adjust_jpeg_quality(images)
        images_rot1 = rotate(images, 30)
        images_rot2 = rotate(images, -30)
        images_rot3 = rotate(images, 60)
        images_rot4 = rotate(images, -60)
        images_rot5 = rotate(images, 120)
        images_rot6 = rotate(images, -120)

        ims = [
            images_identity,
            images_fliplr,
            images_flipud,
            images_crop_and_resize,
            images_brightness,
            images_contrast,
            images_gamma,
            images_jpeg,
            images_rot1,
            images_rot2,
            images_rot3,
            images_rot4,
            images_rot5,
            images_rot6,
        ]
        output = [
            "",
            "_fliplr",
            "_flipud",
            "_crop",
            "_bright",
            "_contrast",
            "_gamma",
            "_jpeg_compress",
            "_rot1",
            "_rot2",
            "_rot3",
            "_rot4",
            "_rot5",
            "_rot6",
        ]

        for i, im in enumerate(ims):
            if not os.path.exists("test"):
                os.makedirs("test")
            images = (sess.run(im) * 255).astype(np.uint8)
            print(images.shape, output[i])
            for j in range(5):
                scipy.misc.imsave("test/" + str(j) + output[i] + ".jpeg", images[j])


if __name__ == "__main__":
    test_augmentation()
