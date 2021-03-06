# 실습 : dropout적용

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

from tensorflow.keras.utils import to_categorical 

y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
print(y)
print(x.shape) #(569, 30)
print(y.shape) # (569, 2) -> reshape됨

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100, validation_data=(x_val, y_val), batch_size=8)

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))

# loss, acc :  0.3076542913913727 0.9561403393745422
# [[7.0065487e-09 1.0000000e+00]
#  [1.0000000e+00 3.3975954e-08]
#  [1.7665387e-06 9.9999821e-01]
#  [3.9296706e-06 9.9999607e-01]]
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]
# [1 0 1 1]

#dropout 후 (loss와 acc 둘 다 떨어지면 성능이 좋아진것, 기준은 loss!)
# loss, acc :  0.22323311865329742 0.9473684430122375
# [[4.8269221e-06 9.9999523e-01]
#  [9.9990499e-01 9.5025200e-05]
#  [7.1617811e-05 9.9992836e-01]
#  [5.7905517e-04 9.9942100e-01]]
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]
# [1 0 1 1]