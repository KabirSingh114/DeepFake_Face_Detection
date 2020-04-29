import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.applications.densenet import DenseNet121


def denseNet(img_width=64, img_height=64, include_top=True):
    model = DenseNet121(include_top=include_top, weights=None, input_shape=(img_width, img_height, 3), classes=2,
    input_tensor=None, pooling=None
    )
    return model