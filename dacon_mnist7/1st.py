import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

def get_tr_data(SEED):    
    tr_data = pd.read_csv("../dacon7/train.csv", index_col = 0).values
    tr_data = shuffle(tr_data, random_state = 77)

    tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
    tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))

    return tr_X, tr_Y


def get_ts_data():
    ts_data = pd.read_csv("./data/test.csv", index_col = 0).values
    ts_X = tf.convert_to_tensor(ts_data[:, 1:], dtype = tf.float32)

    return ts_X

IMAGE_SIZE = [28, 28]
RESIZED_IMAGE_SIZE = [256, 256]

@tf.function
def _reshape_and_resize_tr(flatten_tensor, label):
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor, label


@tf.function
def _reshape_and_resize_ts(flatten_tensor): # without label
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor