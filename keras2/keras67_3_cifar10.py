# 실습. cifar10에 flow 사용. fit generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,BatchNormalization, Activation
from keras.optimizers import Adam


from tensorflow.keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

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

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size= 0.8, random_state=42
)

batch_size=500
train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size, seed=42)
val_gen = test_datagen.flow(x_val, y_val, batch_size=batch_size)
test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

print(train_gen[0][0].shape)
print(val_gen[0][0].shape)
print(test_gen[0][0].shape)
# (40000, 32, 32, 3)
# (10000, 32, 32, 3)
# (10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
print(y_train.shape,y_test.shape,y_val.shape) #(40000, 10) (10000, 10) (10000,10)

model = Sequential()
model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.5, mode='auto')

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=Adam(learning_rate=0.002))

model.fit_generator(
    train_gen, 
    steps_per_epoch=40000/batch_size,
    epochs=1000,
    callbacks=[lr,es],
    validation_data=val_gen
)

loss, acc = model.evaluate_generator(test_gen)
print('loss, acc : ', loss, acc)

# WARNING:tensorflow:From c:\Study\keras2\keras67_3_cifar10.py:90: Model.evaluate_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use Model.evaluate, which supports generators.
# loss, acc :  373.9224853515625 0.04390000179409981