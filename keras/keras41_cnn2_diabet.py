# 실습 : dropout적용

# 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) # (442, 10) (442,)

print(np.max(x),np.min(y)) 
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(500,3, padding='same', input_shape=(10,1,1)))
# model.add(MaxPooling2D(pool_size=2)) 이미지를 분류할 때는 쓰는게 좋지만, 아니면 안쓰는 게 좋다. 
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.fit(x_train, y_train, validation_split=0.2, callbacks=[es], epochs= 300, batch_size=32)

#3. 평가, 예측

loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : " , r2)

# 전처리 전
# loss, mae :  3768.1259765625 49.35535430908203
# RMSE :  61.385062871563136
# R2 :  0.4193986921311813

# 전처리 후 x = x/711.
# loss, mae :  5818.3115234375 65.9879379272461
# RMSE :  76.27786132701927
# R2 :  0.10350140045743106

# MinMaxScaler 통째로 전처리
# loss, mae :  3742.401611328125 49.28392028808594
# RMSE :  61.1751674463525
# R2 :  0.4233624319656353

# 제대로 전처리 (validation_split)
# loss, mae :  3736.541748046875 48.8818359375
# RMSE :  61.12725628077463
# R2 :  0.42426530025513365

#  제대로 전처리(validation_data)
# loss, mae :  3873.400634765625 51.55214309692383
# RMSE :  62.23664736125453
# R2 :  0.4031777867820261

# early stopping
# loss, mae :  3433.62353515625 48.48197937011719
# RMSE :  58.59713822916717
# R2 :  0.4709394090660546

#dropout 후 (성능 좋아짐)
# loss, mae :  3376.09375 48.167259216308594
# RMSE :  58.10415494963148
# R2 :  0.4798040358943778

# cnn
# loss, mae :  3268.77880859375 47.29360580444336
# RMSE :  57.17323338354297
# R2 :  0.4963392498969358