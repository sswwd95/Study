# 행이 다른 앙상블 모델

import numpy as np
from numpy import array

#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],
             [2,3,4],[3,4,5],[4,5,6]])

y1= array([4,5,6,7,8,9,10,11,12,13])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = array([55,65,75])
x2_pred = array([65,75,85])

print('x1.shape : ',x1.shape) # (10,3)
print('x2.shape : ',x2.shape) # (13,3)
print('y1.shape : ', y1.shape) # (10,)
print('y2.shape : ', y2.shape) # (13,)

x1 = x1.reshape(x1.shape[0],x1.shape[1],1) 
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) 

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


output1 = Dense(10)(middle1)
output1 = Dense(1)(output1)

output2 = Dense(10)(middle1)
output2 = Dense(1)(output2)


model = Model(inputs=[input1, input2],
              outputs = [output1,output2])

model.summary()

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode = 'min' )
model.fit([x1, x2], [y1,y2], callbacks=[early_stopping], epochs=1000, batch_size=8)

# 4. 평가, 예측

loss = model.evaluate([x1,x2], [y1,y2], batch_size=8)
print("loss : ", loss)

x1_pred = x1_pred.reshape(1,3,1) 
x2_pred = x2_pred.reshape(1,3) # 덴스로 했으면 덴스 차원 맞춰주기 

result = model.predict([x1_pred, x2_pred])
print('result : ', result)

# 앙상블에서는 행의 크기를 맞춰줘야한다. 

# ValueError: Data cardinality is ambiguous:
#   x sizes: 10, 13
#   y sizes: 10, 13
# Please provide data which shares the same first dimension.