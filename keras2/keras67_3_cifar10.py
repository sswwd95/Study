# 실습. cifar10에 flow 사용. fit generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout


from tensorflow.keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 전처리개념
    horizontal_flip=True,  # True하면 가로로 반전
    vertical_flip=True, # True하면 세로로 반전
    width_shift_range=0.1, # 수평이동
    height_shift_range=0.1, # 수직이동
    rotation_range=5, # 회전
    zoom_range=1.2, # 확대, 축소
    shear_range=0.7, # 크게 하면 더 비스듬하게 찌그러진 이미지가 된다. 
    fill_mode='nearest' # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식.  
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_gen = train_datagen.flow(x_train, y_train, batch_size=16, seed=42)
test_gen = test_datagen.flow(x_test, y_test, shuffle=False)
