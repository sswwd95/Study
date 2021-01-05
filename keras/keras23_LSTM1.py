#1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print("x.shape : ", x.shape) #(4, 3)
print("y.shape : ", y.shape) #(4,)

#1개씩 잘라서 작업하기 위해 reshape. reshape해도 원소의 개수는 동일하다.
x = x.reshape(4, 3, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7]) #(3,) -> reshape해도 원소는 안바뀌고 LSTM에 쓸 수 있게 바꾸는것
x_pred = x_pred.reshape(1, 3 , 1) # -> 와꾸 맞춰준것. 

result = model.predict(x_pred)
print(result)

# 0.0016605528071522713
# [[7.8879824]]
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 20)                420
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 21
=================================================================
'''
# LSTM의 param 480?
# 4(input_dim + num_unit + 1(bias)) * num_unit