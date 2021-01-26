
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

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(100, 2, activation='relu', input_shape=(30,1)))
model.add(Flatten())
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

model.fit(x_train,y_train, epochs=1000, callbacks=[early_stopping], validation_split=0.2, batch_size=8)

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=0))


# LSTM 
# loss, acc :  0.0669548511505127 0.9824561476707458
# [[9.9833578e-01]
#  [4.7121444e-05]
#  [9.9601150e-01]
#  [9.9059403e-01]]
# [1 0 1 1]
# [0]

#Conv1D
# loss, acc :  0.1725590080022812 0.9649122953414917
# [[9.9999785e-01]
#  [6.8931520e-04]
#  [9.9985230e-01]
#  [9.9871457e-01]]
# [1 0 1 1]
# [0]
