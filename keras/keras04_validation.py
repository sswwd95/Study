# 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense 

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

# 교과서 위주로 하는 것 보다는 모의고사 풀면서 하는 것이 좋음. 성능 향상을 위해 훈련을 하면서 검증 데이터도 분리함. 훈련하면서 검증하고를 반복하면서 성능 더 좋아짐
# train, test, validation, predict  -> predict는 예측하는 것. y가 없다. y가 알고싶어서 훈련시키는 것. 


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

model.fit(x_train, y_train, epochs=100, batch_size=1,
         validation_data=(x_validation, y_validation))

#4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

# result = model.predict([9])
result = model.predict(x_train)
print("result : ", result)



