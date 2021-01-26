import numpy as np
#1. 데이터
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],7*7,16)/255.
x_test = x_test.reshape(x_test.shape[0],7*7,16)/255.

print(x_train.shape) #(60000, 49, 16)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(50,2, activation='relu', input_shape=(7*7,16)))
model.add(Conv1D(25,2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience= 20, mode='max')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[es])

#4. 평가, 지표
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc : ', loss, acc)

y_pred = model.predict(x_test)

# cnn
# loss, acc :  0.43148916959762573 0.8651000261306763

# dnn
# loss, acc :  0.5104688405990601 0.8884000182151794

# lstm
# loss, acc :  0.36758238077163696 0.8677999973297119

# conv1d
# loss, acc :  1.2407662868499756 0.879800021648407