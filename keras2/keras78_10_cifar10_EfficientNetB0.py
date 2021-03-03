import numpy as np

# 1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

EfficientNetB0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))
EfficientNetB0.trainable = False

model = Sequential()
model.add(EfficientNetB0)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=10, mode='max')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es],batch_size=16)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

################################### trainable = False ######################################
# VGG16

# VGG19

# Xception

# ResNet50

# ResNet101

# InceptionV3
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)`

# InceptionResNetV2
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)`

# DenseNet121

# MobileNetV2
# loss, acc :  7.910558700561523 0.28610000014305115

# NASNetMobile
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).

# EfficientNetB0
# loss, acc :  2.302685022354126 0.10000000149011612

########################################################################################

################################### trainable = True ######################################

# EfficientNetB0
# loss, acc :  5.556640148162842 0.14030000567436218