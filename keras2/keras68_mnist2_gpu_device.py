
import numpy as np
from tensorflow.keras.datasets import mnist

##########################################################################

# gpus[1]로 이 파일 돌리고 gpus[0]으로 tf17_cnn3_gpu_device.py 파일 돌리면 동시에 같이 돌릴 수 있다
# 동시에 gpu 두 개 쓰는 것 보여준다(난 gpu 1장이라 못해)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus : 
    try:
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
    except RuntimeError as e:
        print(e)
###########################################################################


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test= x_test.reshape(10000,28,28,1)/255. 

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNomalization, Activation
from tensorflow.keras.regularizers import l1,l2,l1_12

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(BatchNomalization()) 
model.add(Activation('relu'))

model.add(Conv2D(32,2, kernel_initializer='he_normal'))
model.add(BatchNomalization())
model.add(Activation('relu'))

model.add(Conv2D(32,2, kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(32,2, strides=2))

model.add(Flatten()) 
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')# filepath :  최저점마다 파일을 생성하는데 파일 안에 그 지점의 w값이 들어간다. predict, evaluate 할 때 파일에서 땡겨쓰면 좋다. 가장 마지막이 제일 값이 좋은 것.
hist = model.fit(x_train,y_train, callbacks=[es], epochs=2, validation_split=0.2, batch_size=16)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747

