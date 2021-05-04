# 오토케라스에 회귀가 먹힐까?
import tensorflow as tf
import numpy as np
import autokeras as ak
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 42
)

print(x_train.shape, x_test.shape)
print(x.shape, y.shape)
# (506, 13) (506,)
'''
#2 . 모델구성
model = ak.StructuredDataRegressor(
    overwrite=True, 
    max_trials=10, 
    loss = 'mse',
    metrics=['mae'],
    directory='C:/data/ak/'
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience=4, verbose=1, restore_best_weights=True, monitor='val_loss', mode='min')
lr = ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
cp = ModelCheckpoint(monitor='val_loss', filepath='C:/data/mc/', save_best_only=True, save_weights_only=True)

model.fit(x_train, y_train, epochs=10) 

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=64)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

model_ak = model.export_model()
try:
    model_ak.save("C:/data/h5/ak_boston", save_format="tf")
except Exception:
    model_ak.save('C:/data/h5/ak_boston.h5')


best_model = model.tuner.get_best_model()
try:
    best_model.save("C:/data/h5/best_ak_boston", save_format="tf")
except Exception:
    best_model.save('C:/data/h5/best_ak_boston.h5')

# 저장 에러날 때
# https://autokeras.com/tutorial/export/

# model3 = load_model('ak_save_boston', custom_objects=ak.CUSTOM_OBJECTS)
# result_boston = model3.evaluate(x_test, y_test)

# y_pred = model3.predict(x_test)
# r2 = r2_score(y_test, y_pred)

# print("load_result :", result_boston, r2)


# ImageRegressor
# ValueError: Expect the data to ImageInput to have shape (batch_size, height, width, channels) or (batch_size, height, width) dimensions, but got input shape [64, 13]

# StructuredDataRegressor
# loss, mae :  15.613632202148438 2.7075181007385254
# RMSE :  3.9514088969530325
# R2 :  0.7870881386715735
'''

from tensorflow.keras.models import load_model
model = load_model('C:/data/h5/ak_boston', custom_objects=ak.CUSTOM_OBJECTS)
model.summary()


best_model = load_model('C:/data/ak/structured_data_regressor/best_model', custom_objects=ak.CUSTOM_OBJECTS)
best_model.summary()
###################################################################

result = model.evaluate(x_test, y_test)
print(result)

best_result = best_model.evaluate(x_test, y_test)
print(best_result)

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 13)]              0
_________________________________________________________________
multi_category_encoding (Mul (None, 13)                0
_________________________________________________________________
normalization (Normalization (None, 13)                27
_________________________________________________________________
dense (Dense)                (None, 128)               1792
_________________________________________________________________
re_lu (ReLU)                 (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128
_________________________________________________________________
re_lu_1 (ReLU)               (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 32)                1056
_________________________________________________________________
re_lu_2 (ReLU)               (None, 32)                0
_________________________________________________________________
regression_head_1 (Dense)    (None, 1)                 33
=================================================================
Total params: 7,036
Trainable params: 7,009
Non-trainable params: 27
_________________________________________________________________
'''
# [15.613630294799805, 2.7075181007385254]