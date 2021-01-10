# cnn으로 구성 , 알아서 2차원에서 4차원으로 늘릴 것. 
# 
# 실습 : dropout적용

import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset=load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape)  # (506, )
print("=================")
print(x[:5]) 
print(y[:10])
print(np.max(x), np.min(x)) # 711.0  0,0
print(dataset.feature_names)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)

print(y_train.shape) # (404, 1)

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(500,3,padding='same',input_shape=(x_train.shape[1],1,1)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.2)) 
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))



# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=20, mode='min') 

model.fit(x_train, y_train, batch_size = 32, callbacks=[es], epochs=2000, validation_split=0.2)


# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

# RMSE, R2 = 회귀모델 지표
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#전처리 전
# loss, mae :  17.961238861083984 3.0568878650665283
# RMSE :  4.238070188962735
# R2 :  0.7825965976870914

# 전처리 후 x = x/711.
# loss, mae :  13.763806343078613 2.824580430984497
# RMSE :  3.7099605406018363
# R2 :  0.8334024435018605

# MinMaxScaler 통째로 전처리
# loss, mae :  13.234770774841309 2.4213950634002686
# RMSE :  3.637962471311643
# R2 :  0.8398059151970972

# 제대로 전처리 (validation_split)
# loss, mae :  11.98676586151123 2.332911729812622
# RMSE :  3.4621908747533503
# R2 :  0.8549118105720956

#  제대로 전처리(validation_data)
# loss, mae :  13.043401718139648 2.84848952293396
# RMSE :  3.6115650362255134
# R2 :  0.8421222507815853

# early stopping
# loss, mae :  11.313929557800293 2.3684475421905518
# RMSE :  3.3636184411913357
# R2 :  0.8630558464213824

# dropout 후 (성능 좋아짐)
# loss, mae :  7.509355068206787 2.066143035888672
# RMSE :  2.7403202713023824
# R2 :  0.9091065280069965

#cnn
# loss, mae :  8.52490520477295 2.2797205448150635
# RMSE :  2.919743971656153
# R2 :  0.8980066370941056