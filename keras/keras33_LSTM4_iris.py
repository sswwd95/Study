# sklearn 데이터셋
# LSTM으로 모델
# Dense와 성능비교
# 다중분류
# 반드시 셔플
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

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)
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

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)


# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(4,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=20, mode = 'max' )

model.fit(x_train, y_train, callbacks=[early_stopping], validation_data=(x_val, y_val), epochs= 100, batch_size=8)

#3. 평가, 예측

loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss, acc : ", loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))
#결과치 나오게 코딩할것.   #argmax

# Dense
'''
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
'''

# LSTM
# loss, acc :  0.12028852850198746 1.0
# [[1.8858841e-15 1.4663759e-03 9.9853361e-01]
#  [9.9953985e-01 4.6012204e-04 1.5635480e-08]
#  [9.8375165e-01 1.6221512e-02 2.6838672e-05]
#  [4.1324918e-07 3.0805749e-01 6.9194210e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]


