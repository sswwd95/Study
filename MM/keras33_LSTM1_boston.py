# sklearn 데이터셋
# LSTM으로 모델
# Dense와 성능비교
# 회귀모델

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
    x, y, train_size = 0.8, random_state = 66
)
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train,
                                                 test_size=0.2, shuffle = True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)


#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(500, activation ='relu', input_shape = (13,1)))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))

model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, callbacks=[early_stopping], batch_size = 32, epochs=1000, validation_data=(x_val,y_val))

# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

# print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#전처리 전
# loss, mae :  27.019460678100586 4.129007816314697
# RMSE :  5.198024668725807
# R2 :  0.6729555986669336

# 전처리 후 x = x/711.
# loss, mae :  13.402970314025879 2.729325771331787
# RMSE :  3.661006663445475
# R2 :  0.8377700310479839

# x 통째로 전처리
# loss, mae :  11.915349960327148 2.335031270980835
# RMSE :  3.4518620227303067
# R2 :  0.8557762106548309

# 제대로 전처리 (validation_split)
#loss, mae :  13.90250301361084 2.6242401599884033
# RMSE :  3.7286061659340923
# R2 :  0.8317236539350162

#  제대로 전처리(validation_data)
# loss, mae :  7.393008708953857 2.1138510704040527
# RMSE :  2.7190087920905297
# R2 :  0.9105147882330157

# LSTM(튜닝 전)
# loss, mae :  87.06122589111328 7.487861633300781
# RMSE :  9.330660513959554
# R2 :  -0.041614812855414796

# LSTM(튜닝 후)
# loss, mae :  10.443477630615234 2.2721614837646484
# RMSE :  3.2316371347232526
# R2 :  0.8750525058832845