import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 각 쉐입에 따라 모델이 달라진다
model = ak.ImageClassifier(
    overwrite=True, # true, false 큰 효과 모르겠다
    max_trials=1, # 최대 시도 2번
    loss = 'mae',
    metrics=['acc']
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

# model.save('C:/data/h5/ak.h5') 안먹힌다
# AttributeError: 'ImageClassifier' object has no attribute 'save'

model2 = model.export_model() # 변수명 바꿔준다
model2.save('C:/data/h5/ak.h5') 