# keras23_LSTM3_scale을 함수형으로 코딩

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

x = x.reshape(13, 3, 1)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,LSTM

input1 = Input(shape = (3,1)) 
dense1 = LSTM(10, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(20)(dense2)
dense4 = Dense(10)(dense3)
outputs = Dense(1)(dense4)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)

x_pred = x_pred.reshape(1, 3 , 1) 

result = model.predict(x_pred)
print(result)

# loss :  0.03733702003955841
# [[80.27483]]

# 함수
# loss :  0.01798980124294758
# [[79.88539]]