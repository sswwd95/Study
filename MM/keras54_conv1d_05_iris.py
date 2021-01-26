
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

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y)
print(x.shape) #(150,4)
print(y.shape) # (150,3) -> reshape됨

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


# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,Flatten

model = Sequential()
model.add(Conv1D(100,3, activation='relu', input_shape=(4,1)))
model.add((Flatten()))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience=20, mode = 'max' )

model.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.2, epochs= 100, batch_size=8)

#3. 평가, 예측

loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss, acc : ", loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))


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

# Conv1D
# loss, acc :  0.08915580809116364 0.9666666388511658
# [[1.5841080e-08 7.8954536e-04 9.9921036e-01]
#  [9.9954802e-01 4.5195059e-04 1.0966091e-10]
#  [9.6257067e-01 3.7429076e-02 2.7399892e-07]
#  [3.2906166e-05 2.3818554e-01 7.6178163e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]


