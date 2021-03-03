import numpy as np
# 1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# print(x_train[0])
# print(x_test[0])
# print(x_train[0].shape) #(32, 32, 3)
# print(np.max(x_train)) #255 -> 이미지에서 특성 가장 큰 건 255= 가장 밝음(빛의 3원색)

x_train = x_train.reshape(50000,16*16,12)/255.
x_test = x_test.reshape(10000,16*16,12)/255.

print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(50000, 10)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(16*16,12)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=3, mode='max')
model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[es],batch_size=32)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=32)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

# cnn
# loss, acc :  1.3659929037094116 0.521399974822998

# dnn
# loss, acc :  1.5606876611709595 0.45190000534057617

# lstm
