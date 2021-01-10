# hist를 이용하여 그래프를 그리시오
# loss, val_loss, acc, val_acc

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
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

## 원핫인코딩 OneHotEncoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(x.shape) #(150,4)
print(y.shape) # (150,3) -> reshape됨

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

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=20, mode = 'max' )

hist = model.fit(x_train, y_train, callbacks=[early_stopping], validation_data=(x_val, y_val), epochs= 100, batch_size=8)

#3. 평가, 예측

loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss, acc : ", loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))
#결과치 나오게 코딩할것.   #argmax

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

# earlystopping 후
# loss, acc :  0.09586071223020554 0.9666666388511658
# [[7.8031324e-09 3.1398889e-04 9.9968600e-01]
#  [9.9978095e-01 2.1909270e-04 1.5728810e-09]
#  [9.7088087e-01 2.9115774e-02 3.4140826e-06]
#  [7.1248389e-05 5.8125854e-01 4.1867021e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 1]

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()


