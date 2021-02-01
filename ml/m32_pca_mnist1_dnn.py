# m31로 만든 0.95이상의 n_component=?를 사용하여 dnn 모델 만들기
# mnist dnn보다 성능 좋게 만들것
# cnn과 비교

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.

print(x_train.shape) #(60000, 784)
print(x_test.shape) #(10000, 784)

pca = PCA(n_components=154)
x2_train = pca.fit_transform(x_train)
x2_test = pca.fit_transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(
    x2_train, y_train, train_size=0.8, random_state=77
)

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(200, activation='relu',input_shape=(154,)))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=20, mode='max')
model.fit(x_train,y_train, callbacks=[es],epochs=100, validation_split=0.2, batch_size=32)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])

# mnist_cnn
# loss, acc :  0.06896616518497467 0.9800999760627747

# mnist_dnn
#loss, acc :  0.13697706162929535 0.9828000068664551

# mnist_pca(0.85이상)
# loss, acc :  0.1388961225748062 0.9789999723434448