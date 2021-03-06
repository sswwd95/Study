# 네이밍 룰
# 파이썬에서 자바 형식으로 네이밍해도 상관없다. 암묵적인 약속일 뿐

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense 
#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])  # w =2 b =0 / w 가중치, b 바이어스  y= wx + b
x_test = np.array([101,102,103,104,105,106,107,108,109,110])  # w=1 b = 10     # x,y 가 다르기 때문에 값이 제대로 안나옴. 
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113])

#2. 모델구성
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(3, input_dim=1, activation='relu'))


model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20)

#4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result = model.predict(x_predict)
print("result : ", result)



