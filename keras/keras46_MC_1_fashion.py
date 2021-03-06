# cnn
import numpy as np

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) #(28,28)
print(np.max) #255

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. # 옛날 방식
x_test = x_test.reshape(10000,28,28,1)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000, 10)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


model = Sequential()
model.add(Conv2D(50,3, padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20,3))
# model.add(MaxPooling2D(pool_size=3))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k46_fashion_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True,mode='auto')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
hist = model.fit(x_train,y_train, epochs=10, validation_split=0.2, batch_size=16, callbacks=[es,cp])

#4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)

y_pred = model.predict(x_test)

# cnn
# loss, acc :  0.43148916959762573 0.8651000261306763


