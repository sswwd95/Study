# LSTM1.py 복사

#1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print("x.shape : ", x.shape) #(4, 3)
print("y.shape : ", y.shape) #(4,)

x = x.reshape(4, 3, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(3,1)))
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

x_pred = np.array([5,6,7]) 
x_pred = x_pred.reshape(1, 3 , 1)  

result = model.predict(x_pred)
print(result)


# LSTM 1
# 0.0016605528071522713
# [[7.8879824]]
'''_______________________________________________________________
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

# SimpleRNN
# 0.0002964430022984743
# [[8.0403805]]
'''
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 20)                420
_________________________________________________________________
'''
# RNN의 param 120?
# num_unit * (num_unit + input_dim + 1(bias))
