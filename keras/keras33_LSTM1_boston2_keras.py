# 텐서플로 데이터셋
# LSTM으로 모델
# Dense와 성능비교
# 회귀모델

import numpy as np

#1. 데이터
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape) # (323, 13)
print(y_train.shape) # (323,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape = (13,1)))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=8, callbacks = [early_stopping], validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test,y_test, batch_size=8)
print('loss, mae : ', loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)


# Dense
# loss, mae :  8.40831184387207 2.143211841583252
# RMSE:  2.899709211991376
# R2:  0.8989917734028398

#LSTM(튜닝 전)
# loss, mae :  28.874502182006836 3.6042473316192627
# RMSE:  5.373500123321262
# R2:  0.6531334856717295

#LSTM(튜닝 후)
