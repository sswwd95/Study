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

from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 전처리 알아서 해 / minmax, train_test_split

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 히든레이어 없어도 괜찮다. 

# 3. 컴파일, 훈련
                # mean_squared_error -> 풀네임도 가능함
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train,y_train, epochs=100, validation_data=(x_val, y_val), batch_size=8)

#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(np.where(y_predict<0.5, 0, 1))
print(y_test[-5:-1])

# loss, acc :  0.4048005938529968 0.9736841917037964
# [[1.0000000e+00]
#  [3.7346189e-04]
#  [9.9999988e-01]
#  [9.9999976e-01]]
# [1 0 1 1]

# np.where사용
# [[1]
#  [0]
#  [1]
#  [1]]
# [1 0 1 1]
