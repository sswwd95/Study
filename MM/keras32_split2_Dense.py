
import numpy as np

a = np. array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=======================")
print(dataset)

x = dataset[:,:4]
y = dataset[:,4:] # [:,-1]
print(x)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
print(y)
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]
print(x.shape) #(6,4)
print(y.shape) #(6,1)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=20, mode='min')

model.fit(x, y, batch_size=1, callbacks=[early_stopping],epochs=1000)

#4. 평가,예측
loss = model.evaluate(x,y, batch_size=1)
print('loss : ',loss)

x_pred = np.array([7,8,9,10])
x_pred = x_pred.reshape(1,4)

result = model.predict(x_pred)
print('result : ',result)

#LSTM
# loss :  0.003044363809749484
# result :  [[10.843571]]

# Dense
# loss :  2.9558577807620168e-12
# result :  [[11.000005]]