import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape) # (150,4)
print(y.shape) # (150,)
print(x[:5])
print(y)

## 원핫인코딩 OneHotEncoding
from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print(x.shape) #(150,4)
print(y.shape) # (150,3) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(4,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs= 100, batch_size=8)

#3. 평가, 예측

loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss, acc : ", loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))

# loss, acc :  0.09609115868806839 0.9666666388511658
# [[1.61500391e-09 3.95920433e-05 9.99960423e-01]
#  [9.99981046e-01 1.89530183e-05 1.12919353e-12]
#  [9.84727979e-01 1.52718639e-02 1.03585734e-07]
#  [7.72912681e-05 4.75085050e-01 5.24837613e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]