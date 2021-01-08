
# 주말과제
# dense 모델로 구성 input_shape=(28*28,)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) #(28,28)
print(np.max)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.

x_trest.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) 

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28*28,)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(9,2))
model.add(Conv2D(8,2))
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=20, mode='max')
model.fit(x_train,y_train, callbacks=[es],epochs=10, validation_split=0.2, batch_size=16)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])

# 실습! 완성하기. 지표는 acc
# 응용 : y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)


# loss, acc :  0.06896616518497467 0.9800999760627747
# [[1.91534921e-09 2.67154132e-10 2.49314354e-07 1.90946957e-05
#   7.72808293e-12 3.49781914e-07 7.58247848e-13 9.99975562e-01
#   1.30290096e-06 3.44504747e-06]
#  [1.30679345e-09 1.61517582e-05 9.99970436e-01 1.33357225e-05
#   9.25153704e-11 2.36759969e-11 1.88991889e-09 2.09380615e-16
#   2.95982296e-08 5.22677013e-09]
#  [5.14019938e-08 9.99820292e-01 5.40180481e-05 7.56906857e-08
#   2.23431962e-05 3.81714790e-06 5.60831609e-07 5.26220429e-05
#   4.60132833e-05 1.95006493e-07]
#  [9.99997020e-01 2.56222702e-15 1.06445093e-07 8.93156313e-11
#   9.58598001e-13 2.58915134e-09 1.23829716e-06 1.28666822e-09
#   4.19732459e-07 1.25077111e-06]
#  [5.45518318e-12 4.83736107e-11 6.29438446e-11 1.22410346e-11
#   9.99997139e-01 3.29958665e-11 1.78054620e-08 3.39257866e-07
#   1.41885250e-08 2.43148202e-06]
#  [6.37071951e-09 9.99483228e-01 8.43201724e-06 4.08888177e-08
#   1.96889305e-05 8.66939800e-08 5.75799053e-09 4.76842863e-04
#   1.15506291e-05 1.22554141e-07]
#  [2.10443774e-16 9.73506609e-10 5.17008381e-11 1.37282123e-05
#   9.56174076e-01 2.04808894e-04 3.62900127e-11 9.85150166e-08
#   2.72487868e-02 1.63584910e-02]
#  [1.04778963e-11 1.80492850e-06 2.15618343e-07 7.25805012e-06
#   6.48551702e-01 1.46446991e-05 7.93345261e-13 1.29903253e-06
#   1.31328437e-09 3.51423025e-01]
#  [6.73193445e-11 1.96291330e-10 7.43669515e-10 1.90695850e-08
#   1.40443120e-07 1.88727379e-01 8.10656190e-01 3.76243828e-12
#   4.68211889e-04 1.48047009e-04]
#  [3.41864218e-16 3.38261049e-13 4.04568360e-13 1.17140075e-07
#   1.21894968e-03 3.29304868e-08 2.13533548e-17 4.16680632e-05
#   1.26934574e-05 9.98726428e-01]]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]