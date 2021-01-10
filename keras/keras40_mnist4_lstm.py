# lstm모델로 구성 input_shape=(28*28,1)
# lstm모델로 구성 input_shape=(28*14,2)
# lstm모델로 구성 input_shape=(28*7,4)
# lstm모델로 구성 input_shape=(7*7,16) 등 숫자 마음대로(위에 숫자 다 해보기)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape) #(28,28)
# print(np.max)

# input_shape=(28*28,1)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],1)/255.
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],1)/255.

# input_shape=(28*14,2)
# x_train = x_train.reshape(x_train.shape[0],28*14,2)/255.
# x_test = x_test.reshape(x_test.shape[0],28*14,2)/255.

x_train = x_train.reshape(x_train.shape[0],7*7,16)/255.
x_test = x_test.reshape(x_test.shape[0],7*7,16)/255.


print(x_train.shape) #(60000, 784, 1)
print(x_test.shape) # (10000, 784, 1)


#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(200, activation='relu',input_shape=(7*7,16)))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.fit(x_train, y_train, callbacks=[es], epochs=5, validation_split=0.2, batch_size=128)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
# print(y_predict)
# print(y_test[:10])

# mnist_cnn
# loss, acc :  0.06896616518497467 0.9800999760627747

# mnist_dnn
# loss, acc :  0.13697706162929535 0.9828000068664551

# mnist_lstm
# lstm모델로 구성 input_shape=(28*28,1) -> 너무 느림..
# lstm모델로 구성 input_shape=(28*14,2) 
# lstm모델로 구성 input_shape=(28*7,4) 
# lstm모델로 구성 input_shape=(7*7,16) -> loss, acc :  0.3901827931404114 0.8531000018119812
