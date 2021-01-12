# k51_1_파일의 fit까지 다 저장. 

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
x_test= x_test.reshape(10000,28,28,1)/255. 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

#2. 모델
from tensorflow.keras.models import Sequential,load_model

model = load_model('../data/h5/k51_1_model2.h5')
# fit 다음에 model save하면 가중치까지 저장된다. 

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
print('model2_loss : ', result[0])
print('model2_acc : ', result[1])

# 모델만 저장
# model1_loss :  0.0803590789437294
# model1_acc :  0.9736999869346619

# 모델과 fit 저장
# model2_loss :  0.0760040208697319
# model2_acc :  0.9757999777793884