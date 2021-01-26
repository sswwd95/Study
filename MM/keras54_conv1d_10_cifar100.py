import numpy as np
# 1. 데이터
from tensorflow.keras.datasets import cifar100
(x_train,y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32*32,3).astype('float32')/255.
x_test = x_test.reshape(10000,32*32,3).astype('float32')/255.

print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(50000, 100)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D,Flatten

model = Sequential()
model.add(Conv1D(64,2, activation='relu', input_shape=(32*32,3)))
model.add(Conv1D(32,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=5, mode='max')
model.fit(x_test, y_test, epochs=10, validation_split=0.2, callbacks=[es],batch_size=64)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=64)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

# cnn
# loss, acc :  2.931725025177002 0.8048999905586243

# dnn
# loss, acc :  3.3800344467163086 0.6758999824523926

# LSTM
# loss, acc :  4.6056694984436035 0.009999999776482582

# Conv1D
# loss, acc :  3.8336269855499268 0.10159999877214432