import numpy as np
from numpy import array

#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],
             [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = array([55,65,75])
x2_pred = array([65,75,85]) # (3,) -> (1,3)dense -> (1,3,1)LSTM

# 실습 : 앙상블 모델(85의 근사치 만들기)

print('x1.shape : ',x1.shape) # (13,3)
print('x2.shape : ',x2.shape) # (13,3)
print('y.shape : ', y.shape) # (13,)

x1 = x1.reshape(x1.shape[0],x1.shape[1],1) 
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) 

# from sklearn.model_selection import train_test_split
# x1_train, x1_test, y_train, y_test, x2_train, x2_test = train_test_split(
#     x1,x2,y, shuffle = True, train_size = 0.8
# )

# 2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(3,1))
dense1 = LSTM(300, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(300, activation='relu')(input2)
dense2 = Dense(200, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)

from tensorflow.keras.layers import concatenate

merge1 = concatenate([dense1, dense2])
middle1 = Dense(300, activation='relu')(merge1)      # middle 없어도 된다. 
middle1 = Dense(100, activation='relu')(middle1)
middle1 = Dense(100, activation='relu')(middle1)
middle1 = Dense(30, activation='relu')(middle1)
output1 = Dense(1)(middle1)

model = Model(inputs=[input1, input2],
              outputs = output1)

model.summary()

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode = 'min' )
model.fit([x1, x2], y, callbacks=[early_stopping], 
           epochs=500, batch_size=8)

# 4. 평가, 예측

loss = model.evaluate([x1,x2], y, batch_size=8)
print("loss : ", loss)

# x_pred = array([55,65,75])
# y_pred = array([65,75,85])

x1_pred = x1_pred.reshape(1,3,1) 
x2_pred = x2_pred.reshape(1,3,1)

result = model.predict([x1_pred, x2_pred])
print('result : ', result)

# loss :  0.014562631025910378
# result :  [[77.819756]]