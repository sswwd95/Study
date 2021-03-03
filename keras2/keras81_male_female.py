# 실습. fit 사용

import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_val = np.load('../data/image/gender/npy/keras67_val_x.npy')
y_val = np.load('../data/image/gender/npy/keras67_val_y.npy')


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (1389, 128, 128, 3) (1389,)
# (347, 128, 128, 3) (347,)
print(x_train[0].shape)
# plt.imshow(x_train[0])
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size=0.8, random_state=7
)

# from tensorflow.keras.applications.vgg19 import preprocess_input
# x_train = preprocess_input(x_train)
# x_test = preprocess_input(x_test)

print(x_train.shape, y_train.shape)
# (1111, 128, 128, 3) (1111,)

from tensorflow.keras.applications import VGG16,VGG19,Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
VGG16.trainable = False

model = Sequential()
model.add(VGG16)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')
filepath = ('../data/modelcheckpoint/vgg16-{val_acc:.4f}.hdf5')
cp = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

history= model.fit(x_train, y_train, epochs=1000, callbacks=[es,rl], validation_data=(x_val, y_val) )
print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

model.save('../data/h5/vgg16.h5')


loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

# loss, acc :  0.966610848903656 0.8571428656578064

# vgg16
# loss, acc :  0.43815216422080994 0.8571428656578064

# VGG19
# loss, acc :  0.6961669325828552 0.7142857313156128

# Xception
# loss, acc :  1.979369044303894 0.5714285969734192
