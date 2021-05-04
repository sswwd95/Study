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
'''
model = ak.ImageClassifier(
    overwrite=True, 
    max_trials=5, 
    loss = 'binary_crossentropy',
    metrics=['acc'],
    directory='C:/data/ak/'
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience=4, verbose=1, restore_best_weights=True, monitor='val_loss', mode='min')
lr = ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
cp = ModelCheckpoint(monitor='val_loss', filepath='C:/data/mc/', save_best_only=True, save_weights_only=True)
 
model.fit(x_train, y_train, epochs=10, validation_split=0.2,
            callbacks=[es,lr,cp]) 

result = model.evaluate(x_test, y_test)

print(result)

model_ak = model.export_model()
model_ak.save('C:/data/h5/ak_cancer.h5') 

best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/best_ak_cancer.h5')

# [0.06950535625219345, 0.9649122953414917]
'''

from tensorflow.keras.models import load_model
model = load_model('C:/data/h5/ak_cancer.h5')
model.summary()


best_model = load_model('C:/data/ak/image_classifier/best_model', custom_objects=ak.CUSTOM_OBJECTS)
best_model.summary()
###################################################################

result = model.evaluate(x_test, y_test)
print(result)

best_result = best_model.evaluate(x_test, y_test)
print(best_result)

'''
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 30, 1)]           0
_________________________________________________________________
cast_to_float32 (CastToFloat (None, 30, 1)             0
_________________________________________________________________
expand_last_dim (ExpandLastD (None, 30, 1, 1)          0
_________________________________________________________________
normalization (Normalization (None, 30, 1, 1)          3
_________________________________________________________________
conv2d (Conv2D)              (None, 30, 1, 32)         320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 1, 64)         18496
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 1, 64)         0
_________________________________________________________________
dropout (Dropout)            (None, 15, 1, 64)         0
_________________________________________________________________
flatten (Flatten)            (None, 960)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 960)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 961
_________________________________________________________________
classification_head_1 (Activ (None, 1)                 0
=================================================================
Total params: 19,780
Trainable params: 19,777
Non-trainable params: 3
_________________________________________________________________

'''

# 4/4 [==============================] - 2s 7ms/step - loss: 0.0695 - acc: 0.9649
# [0.06950535625219345, 0.9649122953414917]
# 4/4 [==============================] - 0s 3ms/step - loss: 0.0695 - acc: 0.9649
# [0.06950535625219345, 0.9649122953414917]
