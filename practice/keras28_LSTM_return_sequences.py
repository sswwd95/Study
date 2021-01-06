# keras23_3을 카피해서 LSTM층을 두 개를 만들 것!

# model.add(LSTM(10,input_shape=(3,1)))
# modle.add(LSTM(10))

import numpy as np
# 1. 데이터
x = np.array([[1,2,3] ,[2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]) 

print("x.shape : ", x.shape) #(13, 3)
print("y.shape : ", y.shape) #(13,)
print(x.shape[0])
print(x.shape[1])

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0],x.shape[1],1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(10, return_sequences = True, activation='relu', input_shape=(3,1)))
model.add(LSTM(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode = 'min' )

model.fit(x,y, callbacks=[early_stopping], epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)

x_pred = x_pred.reshape(1,3,1) 

result = model.predict(x_pred)
print(result)

# LSTM 1개
# loss :  0.03733702003955841
# [[80.27483]]

# LSTM 2개
# loss :  0.006023036781698465
# [[81.46263]]

# LSTM 3개
# loss :  0.044297657907009125
# [[82.34313]]

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 10)                840
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
'''