# 네이밍 룰
# 파이썬에서 자바 형식으로 네이밍해도 상관없다. 암묵적인 약속일 뿐
# 카멜케이스 

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense 
#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])
#데이터를 두개로 나눴지만 실제로 데이터는 1개임. 원래 데이터는 1~8까지인 것.


#2. 모델구성
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1700, batch_size=1)

#4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1) 
print('loss : ', loss)

result = model.predict([9])  
print("result : ", result)



