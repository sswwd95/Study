#실습 : 19_1,2,3,4,5 Early stopping까지 총 6개의 파일 완성하기

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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(10, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
         callbacks=[early_stopping], epochs= 300, batch_size=8)

#3. 평가, 예측

loss,mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : " , r2)

# 1번
# loss, mae :  3768.1259765625 49.35535430908203
# RMSE :  61.385062871563136
# R2 :  0.4193986921311813

# 2번 
# loss, mae :  5818.3115234375 65.9879379272461
# RMSE :  76.27786132701927
# R2 :  0.10350140045743106

# 3번
# loss, mae :  3742.401611328125 49.28392028808594
# RMSE :  61.1751674463525
# R2 :  0.4233624319656353

#4번 
# loss, mae :  3736.541748046875 48.8818359375
# RMSE :  61.12725628077463
# R2 :  0.42426530025513365

#5번
# loss, mae :  3873.400634765625 51.55214309692383
# RMSE :  62.23664736125453
# R2 :  0.4031777867820261

#6번
# loss, mae :  3433.62353515625 48.48197937011719
# RMSE :  58.59713822916717
# R2 :  0.4709394090660546