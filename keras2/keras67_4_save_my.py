# 실습. fit_generator 사용

import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# male = 841
# female = 895

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255,
    validation_split=0.2  
)

batch_size = 32
train_gen=datagen.flow_from_directory(
    'c:/data/image/gender/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=batch_size,
    subset="training"
)

val_gen=datagen.flow_from_directory(
    'c:/data/image/gender/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=batch_size,
    subset="validation"
)



print(train_gen[0][0].shape, val_gen[0][0].shape)
# (1389, 128, 128, 3) (347, 128, 128, 3)


np.save('../data/image/gender/npy/keras67_train_x.npy', arr=train_gen[0][0])
np.save('../data/image/gender/npy/keras67_train_y.npy', arr=train_gen[0][1])
np.save('../data/image/gender/npy/keras67_val_x.npy', arr=val_gen[0][0])
np.save('../data/image/gender/npy/keras67_val_y.npy', arr=val_gen[0][1])