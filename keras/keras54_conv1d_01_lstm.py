# conv1d
# LSTM과 비교

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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D

model = Sequential()
model.add(Conv1D(128, 2, input_shape=(3,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/k54_1_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, batch_size = 8, callbacks=[early_stopping, cp], epochs=1000, validation_split=0.2)

# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)

x_pred = x_pred.reshape(1,3,1) 

y_predict = model.predict(x_pred)


# lstm
# loss :  0.03733702003955841

#conv1d
# loss, mae :  0.38369080424308777 0.5462395548820496

# summary
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 2, 128)            384
_________________________________________________________________
dense (Dense)                (None, 2, 64)             8256
_________________________________________________________________
dense_1 (Dense)              (None, 2, 32)             2080
_________________________________________________________________
dense_2 (Dense)              (None, 2, 16)             528
_________________________________________________________________
dense_3 (Dense)              (None, 2, 8)              136
_________________________________________________________________
dense_4 (Dense)              (None, 2, 1)              9
=================================================================
Total params: 11,393
Trainable params: 11,393
Non-trainable params: 0
'''