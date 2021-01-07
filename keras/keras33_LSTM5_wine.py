# sklearn 데이터셋
# LSTM으로 모델
# Dense와 성능비교
# 다중분류

from sklearn.datasets import load_wine
import numpy as np

dataset = load_wine()
print(dataset.DESCR)
# :Number of Instances: 178 (50 in each of three classes)
# :Number of Attributes: 13 numeric, predictive attributes and the class 
# 178헹, 13열
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape) #(178,13)
print(y.shape) #(178,)

# 실습, DNN

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
print(y.shape) #(178,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 55
)

x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                test_size = 0.2, shuffle = True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(300, activation = 'relu', input_shape=(13,1)))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 20, mode='max')

model.fit(x_train, y_train, callbacks=[early_stopping], validation_data=(x_val,y_val), batch_size=8, epochs=1000)

#. 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss,acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(np.argmax(y_predict,axis=-1))


# Dense
# loss, acc :  0.00012851390056312084 1.0
# [[8.1283273e-09 1.0000000e+00 3.7419234e-10]
#  [2.8588536e-07 9.9999976e-01 2.1940119e-08]
#  [2.6080270e-06 9.9999702e-01 4.0747332e-07]
#  [1.3668525e-09 2.1839366e-09 1.0000000e+00]]
# [1 1 1 2]

# LSTM
# loss, acc :  0.20736928284168243 0.9166666865348816
# [[3.1833672e-03 9.2163265e-01 7.5184017e-02]
#  [7.1840012e-05 1.5121807e-01 8.4871006e-01]
#  [6.3422293e-04 3.8524845e-01 6.1411732e-01]
#  [1.8934759e-05 1.0555633e-01 8.9442468e-01]]
# [1 2 2 2]