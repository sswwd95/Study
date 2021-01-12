import numpy as np

x_train= np.load('../data/npy/i_x_train.npy')
x_test= np.load('../data/npy/i_x_test.npy')
y_train= np.load('../data/npy/i_y_train.npy')
y_test= np.load('../data/npy/i_y_test.npy')

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(50000, 100)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(100,2, padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(90,2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/k50_cifar100_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, mode='auto', save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='acc', patience=10, mode='max')
model.fit(x_test, y_test, epochs=50, validation_split=0.2, callbacks=[es,cp],batch_size=32)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=32)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

