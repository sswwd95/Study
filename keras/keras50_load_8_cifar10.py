import numpy as np

x_train= np.load('../data/npy/c_x_train.npy')
x_test= np.load('../data/npy/c_x_test.npy')
y_train= np.load('../data/npy/c_y_train.npy')
y_test= np.load('../data/npy/c_y_test.npy')

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(50000, 10)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(20,3,padding='same', input_shape=(32,32,3)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k50_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monito='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=3, mode='min')
model.fit(x_test, y_test, epochs=10, validation_split=0.2, callbacks=[es,cp],batch_size=16)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)
