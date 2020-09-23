# Utility

import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize

from skimage.transform import resize
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from PIL import Image
from functools import reduce
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=798, crop_width=1064, flag = True):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    ny = 0
    nx = 0
    for path in paths:
        image = imread(path, mode='L')
        #image = resize(image, (crop_height, crop_width), 'nearest', mode='L')

        if flag:
            image = np.stack(image, axis=0)
            image = np.stack((image, image, image), axis=-1)
        else:
            image = np.stack(image, axis=0)
            image = np.stack(image, axis=-1)

        images.append(image)
    images = np.stack(images, axis=-1)

    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='RGB')

        if height is not None and width is not None:
            image = resize(image, [height, width], interp='nearest')

        images.append(image)

    images = np.stack(images, axis=0)
    print('images shape gen:', images.shape)
    return images


def save_images(paths, datas, save_path, udata, vdata, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)

        yuv_img = np.stack((data, udata, vdata), axis=-1)
        #clr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        clr_img = YUV2RGB(yuv_img)

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)


        # new_im = Image.fromarray(data)
        # new_im.show()

        imsave(path, clr_img)

def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304

    return rgb