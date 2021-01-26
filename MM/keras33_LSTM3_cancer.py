# sklearn 데이터셋
# LSTM으로 모델
# Dense와 성능비교
# 이진분류

# 유방암 예측 모델
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape)  # (569,) 
# print(x[:5])
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, shuffle=True)

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
model.add(LSTM(100, activation='relu', input_shape=(30,1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=20, mode = 'max' )

model.fit(x_train,y_train, epochs=1000, callbacks=[early_stopping], validation_data=(x_val, y_val), batch_size=8)

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=0))

#Dense
'''
# loss, acc :  0.4048005938529968 0.9736841917037964
# [[1.0000000e+00]
#  [3.7346189e-04]
#  [9.9999988e-01]
#  [9.9999976e-01]]
# [1 0 1 1]
# [0]

# earlystopping 후
# loss, acc :  0.14489495754241943 0.9736841917037964
# [[0.9999343 ]
#  [0.00247921]
#  [0.99968374]
#  [0.9996431 ]]
# [1 0 1 1]
# [0]
'''

#LSTM
# loss, acc :  0.2604004144668579 0.8947368264198303
# [[0.9990983 ]
#  [0.20188546]
#  [0.9989675 ]
#  [0.98901504]]
# [1 0 1 1]
# [0]

# LSTM 튜닝 후
# loss, acc :  0.0669548511505127 0.9824561476707458
# [[9.9833578e-01]
#  [4.7121444e-05]
#  [9.9601150e-01]
#  [9.9059403e-01]]
# [1 0 1 1]
# [0]

