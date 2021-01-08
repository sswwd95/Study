
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
# x_train의 최댓값 : 255
# .astype('float32') -> 정수형을 실수형으로 바꾸는 것
x_test= x_test.reshape(10000,28,28,1)/255. 
# 이렇게 해도 실수형으로 바로 된다. 
#(x_trest.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) -> 코딩할 때 이렇게 쓰기!

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(9,2))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=20, mode='max')
model.fit(x_train,y_train, callbacks=[es],epochs=100, validation_split=0.2, batch_size=64)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])

# 실습! 완성하기. 지표는 acc
# 응용 : y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)

# loss, acc :  0.24955864250659943 0.9714999794960022
# [[1.45571898e-22 3.43619074e-15 1.95033042e-16 1.24952335e-13
#   2.37647799e-15 1.67261917e-19 1.88275892e-26 1.00000000e+00
#   4.08464036e-16 2.67367390e-08]
#  [1.94545588e-16 1.97452799e-13 1.00000000e+00 2.21247395e-11
#   2.40233180e-19 1.41252419e-16 6.00700603e-23 6.73841603e-16
#   3.60017354e-13 9.45158004e-29]
#  [5.02280952e-25 1.00000000e+00 2.98105487e-23 2.25338397e-19
#   6.37647144e-16 6.73670804e-27 2.92959433e-29 1.43234674e-22
#   8.87916434e-24 1.40855459e-29]
#  [1.00000000e+00 3.36757914e-32 2.33994546e-16 9.08773013e-21
#   7.36832216e-27 2.45297722e-19 1.72873736e-20 7.13391366e-28
#   7.72618654e-25 1.44830459e-19]
#  [2.27511396e-13 3.79153279e-13 1.95335894e-14 1.08527546e-13
#   1.00000000e+00 2.58542806e-28 9.54762717e-18 1.38932920e-19
#   9.40681254e-26 7.15823720e-11]
#  [1.58038398e-31 1.00000000e+00 1.68252971e-29 1.58647205e-24
#   8.21450125e-20 2.97793540e-34 3.03628307e-37 5.16146818e-28
#   2.26769468e-30 2.57823626e-37]
#  [4.78979877e-11 8.00634697e-11 7.35964241e-12 2.82764472e-11
#   1.00000000e+00 1.00262004e-23 3.10510971e-15 8.08802803e-16
#   6.12362389e-20 1.67759371e-08]
#  [5.06160222e-06 4.78558177e-05 1.84002900e-04 1.96929555e-02
#   2.92712997e-04 2.97558261e-04 9.22658364e-06 2.90618627e-03
#   8.35282821e-03 9.68211651e-01]
#  [9.84690800e-08 2.35193738e-06 7.91752264e-09 2.39327203e-09
#   5.61518448e-07 9.29169687e-07 9.99994993e-01 1.66302030e-07
#   7.96688482e-07 1.17872236e-08]
#  [8.88520417e-22 8.28616201e-13 1.10099857e-16 2.19326432e-11
#   7.17187629e-11 7.80842676e-13 2.50325053e-17 2.60157429e-09
#   1.07958629e-08 1.00000000e+00]]
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
