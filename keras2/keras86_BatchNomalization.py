
import numpy as np
from tensorflow.keras.datasets import mnist

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
# regularizers :정칙화, 정규화? 

# minmax 정규화, standard 일반화? 
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(BatchNomalization()) # 정규화, 일반화? 
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

# layer상에서 한정짓는 것들 정리
'''
kernel_initializer : He        relu친구들?
(가중치 초기화)       Xavier    sigmoid, tanh?

bias_initializer 
kernel_regularizer

batchNormalization
Dropout


batchNormalization과 Dropout은 통상적으로 같이 쓰면 안된다고 한다. 성능에 중복적인 부분이 있다
근데 gan에서는 같이쓴다. 명확히 좋다는 것은 없음





kernel_initializer : He - relu, selu, elu ....
                     Xavier - sigmoid, tahn
(kernel : 가중치(weight))
                     kernel_initializer 를 하게 되면 얼마나 gradient 를 잘 전달 할 수 있느냐와
                     layer 를 얼마나 깊게 쌓을 수 있느냐가 정해짐
                     kernel_initializer 에 존재하는 he 와 xavier 는 각각
                     relu, selu, elu 등과 sigmoid, tahn 등에 사용할 때 적합하다

bias_initializer : bias 는 활성화 함수에 직접적으로 관여하게 되므로 몹시 중요한데,
                   기존에는 0.01 이나 0.1 처럼 매우 작은 양수를 주었으나,
                   학습 방법이 개선 된 지금은 보통 0 으로 초기화를 시킴

kernel_regularizer : 레이어 복잡도에 제한을 두어 가중치가 가장 작은 값을 가지도록 강제함
                     (가중치 값의 분포가 균일해짐)

BatchNormalization : 레이어에 들어가는 batch 값들을 정규화 시킴

Dropout : 훈련 할 때 node 의 갯수를 무작위로 줄임 / 검증할 때엔 dropout 을 하지 않음

Batch, Dropout 과 같이 쓰면 안 좋다고는 하지만 무조건 확정적인 것은 아니며,
실제로도 gan 에서도 함께 쓰이기도 한다
'''