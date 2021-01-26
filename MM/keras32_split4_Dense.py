# 과제 및 실습 LSTM
# 전처리 얼리스탑핑 등등 다 넣을 것
# 데이터는 1~100 

#   x         y
# 1,2,3,4,5   6
# ...
# ...
# 95,96,97,98,99 100

# #predcit를 만들것
# #96,97,98,99,100-> 101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101,102,103,104,105)
# LSTM과 결과 비교!

import numpy as np

a = np. array(range(1,101))
size = 6

b = np.array(range(96,106))
size_pred = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
predict = split_x(b, size_pred)

print("=======================")
print(dataset)

x = dataset[:,:5]
y = dataset[:,5:] # [:,-1]
x_pred = predict[:,:5]

print(x.shape) #(95,5)
print(y.shape) #(95,1)
print(x_pred.shape) #(5,5)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# print(x_train.shape)  #(76, 5, 1)
# print(x_test.shape)  #(19, 5, 1)
# print(x_pred.shape)  #(5, 5, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(5,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=20, mode='min')

model.fit(x_train, y_train, batch_size=1, callbacks=[early_stopping],epochs=1000,validation_split=0.2)

#4. 평가,예측
loss = model.evaluate(x_test,y_test, batch_size=1)
print('loss : ',loss)

# x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)

result = model.predict(x_pred)
print('result : ',result)

# LSTM
# loss :  0.04851379618048668
# result :  [[101.414085]
#  [102.41297 ]
#  [103.41383 ]
#  [104.416824]
#  [105.422066]]

# Dense
# loss :  0.004237275570631027
# result :  [[100.94791 ]
#  [101.94839 ]
#  [102.94887 ]
#  [103.949326]
#  [104.94979 ]]