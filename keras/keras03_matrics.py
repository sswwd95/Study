# 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델구성
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))                                 

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# metrcis 평가지표. 검증셋과 연관. 훈련 과정을 모니터링 하는데 사용

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

# result = model.predict([9])
result = model.predict(x_train)
print("result : ", result)



