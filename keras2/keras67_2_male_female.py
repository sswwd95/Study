# 실습. fit 사용

import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_val = np.load('../data/image/gender/npy/keras67_val_x.npy')
y_val = np.load('../data/image/gender/npy/keras67_val_y.npy')


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (1389, 128, 128, 3) (1389,)
# (347, 128, 128, 3) (347,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size=0.8, random_state=7
)

print(x_train.shape, y_train.shape)
# (1111, 128, 128, 3) (1111,)

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
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

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')

history= model.fit(x_train, y_train, epochs=1000, callbacks=[es,rl], validation_data=(x_val, y_val) )
print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

# loss :  0.00012983998749405146
# acc :  1.0
# 1/1 [==============================] - 0s 998us/step - loss: 0.9666 - acc: 0.8571
# loss, acc :  0.966610848903656 0.8571428656578064
