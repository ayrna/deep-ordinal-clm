from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool
from skimage.io import imread
from functools import partial
from itertools import repeat
import numpy as np
import os
import random

def generate_random_augmentation(p, shape):
    aug = {}

    if 'rotation_range' in p:
        aug['theta'] = random.uniform(-p['rotation_range'], p['rotation_range'])

    if 'width_shift_range' in p:
        aug['ty'] = random.uniform(-p['width_shift_range'] * shape[1], p['width_shift_range'] * shape[1])

    if 'height_shift_range' in p:
        aug['tx'] = random.uniform(-p['height_shift_range'] * shape[0], p['height_shift_range'] * shape[0])

    if 'shear_range' in p:
        aug['shear'] = random.uniform(-p['shear_range'], p['shear_range'])

    if 'zoom_range' in p:
        aug['zy'] = aug['zx'] = random.uniform(1 - p['zoom_range'], 1 + p['zoom_range'])

    if 'flip_horizontal' in p:
        aug['flip_horizontal'] = p['flip_horizontagenerate_random_augmentationl']

    if 'flip_vertical' in p:
        aug['flip_vertical'] = p['flip_vertical']

    if 'channel_shift_range' in p:
        aug['channel_shift_intencity'] = random.uniform(-p['channel_shift_range'], p['channel_shift_range'])

    if 'brightness_range' in p:
        aug['brightness'] = random.uniform(-p['brightness'], p['brightness'])

    return aug

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

# Process a single image
def process_data(augmentation, x):
    # Apply data augmentation
    if len(augmentation) > 0:
        if 'crop' in augmentation:
            x = random_crop(x, (augmentation['crop'], augmentation['crop']))

        x = ImageDataGenerator().apply_transform(x, generate_random_augmentation(augmentation, shape=x.shape))

    return x

class SmallGenerator(Sequence):
    def __init__(self, x, y, num_classes, mean=None, std=None, batch_size=128, augmentation={}, workers=7, one_hot=True):
        self._x = x
        self._y = y
        self._num_classes = num_classes
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._one_hot = one_hot
        
        super(SmallGenerator, self).__init__()

    def __len__(self):
        return int(np.ceil(len(self._x) / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        # batch_x = self._p.starmap(process_data, zip(repeat(self._augmentation), (batch_x[i, :, :, :] for i in range(batch_x.shape[0]))))
        func = partial(process_data, self._augmentation)
        batch_x = np.array(self._p.map(func, batch_x))

        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._one_hot:
            batch_y = to_categorical(batch_y, num_classes=self._num_classes)

        return np.array(batch_x), np.array(batch_y)

    def __del__(self):
        if self._p is not None:
            self._p.close()
            self._p.terminate()
            self._p.join()


def process_data_path(augmentation, force_rgb, base_path, path):
    img = imread(os.path.join(base_path, path))

    # Convert to RGB if grayscale
    if force_rgb and len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)

    # Apply data augmentation
    if len(augmentation) > 0:
        img = ImageDataGenerator().apply_transform(img, generate_random_augmentation(augmentation, shape=img.shape))

    return img

class BigGenerator(Sequence):
    def __init__(self, df, base_path, num_classes, x_col='x', y_col='y', mean=None, std=None, batch_size=128, augmentation={}, workers=7, one_hot=True, force_rgb=True):
        self._df = df
        self._base_path = base_path
        self._num_classes = num_classes
        self._x_col = x_col
        self._y_col = y_col
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._one_hot = one_hot
        self._force_rgb = force_rgb

        super(BigGenerator, self).__init__()

    def __len__(self):
        return int(np.ceil(self._df.shape[0] / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_paths = self._df.iloc[idx * self._batch_size : (idx + 1) * self._batch_size][self._x_col]
        batch_y = self._df.iloc[idx * self._batch_size : (idx + 1) * self._batch_size][self._y_col]

        # Load batch images using multiprocessing
        func = partial(process_data_path, self._augmentation, self._force_rgb, self._base_path)
        batch_x = np.array(self._p.map(func, batch_paths))

        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._one_hot:
            batch_y = to_categorical(batch_y, num_classes=self._num_classes)

        return np.array(batch_x), np.array(batch_y)

    def __del__(self):
        if self._p is not None:
            self._p.close()
            self._p.terminate()
            self._p.join()