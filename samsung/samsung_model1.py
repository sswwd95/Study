import numpy as np
from  tensorflow.keras.models import load_model

samsung = np.load('../data/npy/samsung.npy')

x_train = samsung