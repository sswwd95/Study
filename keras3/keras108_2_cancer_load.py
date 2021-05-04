# 오토케라스에 이진분류가 먹힐까?

import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
print(x_train.shape, x_test.shape)

# (455, 30, 1) (114, 30, 1)

# 각 쉐입에 따라 모델이 달라진다
model = ak.ImageClassifier(
    overwrite=True, # true, false 큰 효과 모르겠다
    max_trials=2, # 최대 시도 2번
    loss = 'binary_crossentropy',
    metrics=['acc'],
    directory='C:/data/ak/'

)
# model.summary() 먹히지 않는다. 오토케라스는 모델을 자동으로 만들어 주는데, 이 부분에서는 모델을 완성해주는 것이 아니라 안먹힌다.

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience=4, verbose=1, restore_best_weights=True, monitor='val_loss', mode='min')
lr = ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
cp = ModelCheckpoint(monitor='val_loss', filepath='C:/data/mc/', save_best_only=True, save_weights_only=True)
 
model.fit(x_train, y_train, epochs=1, validation_split=0.2,
            callbacks=[es,lr,cp]) 
# 디폴트 val_split=0.2로 먹혀준다

result = model.evaluate(x_test, y_test)

print(result)

model_ak = model.export_model()
model_ak.save('C:/data/h5/ak_cancer.h5') 

best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/best_ak_cancer.h5')

# [0.5197036862373352, 0.8684210777282715]