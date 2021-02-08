import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Input

datasets = load_iris()
x = datasets.data
y = datasets.target

#1. 데이터 / 전처리
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


#2. 모델
def build_model(drop=0.5, optimizer='adam', act = 'relu', node = 512):
    inputs = Input(shape = (x_train.shape[1],), name='input')
    x = Dense(node,activation=act, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node*0.5,activation=act, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node*0.25,activation=act, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3,activation=act, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'],loss='mse')
    return model

def create_hyperparameters():
    batches = [16,32,64]
    optimizers = ['rnsprop','adam']
    dropout = [0.1,0.2,0.3]
    act = ['relu']
    node = [128,64]
    return{'batch_size' : batches, 'optimizer' : optimizers, 'drop':dropout, 'node':node, 'act':act}
hyperparameters = create_hyperparameters()
model2 = build_model()

#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose=1)
################ 함수형 모델을 랩핑해야한다. #############################

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k62_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.5,verbose=1)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2,hyperparameters,cv=2)
search = GridSearchCV(model2,hyperparameters,cv=2)

search.fit(x_train, y_train, verbose=1,validation_split=0.2, epochs=100, callbacks = [es, lr, cp])
print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것

print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것

print(search.best_score_) # acc스코어와는 다르다. 

acc = search.score(x_test,y_test)
print('최종 스코어 : ',acc)

# RandomizedSearchCV
# {'optimizer': 'adam', 'node': 128, 'drop': 0.1, 'batch_size': 16, 'act': 'elu'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001B3FB3436A0>
# -80.11432266235352
# 7/7 [==============================] - 0s 1ms/step - loss: 97.8084 - mae: 6.8656
# 최종 스코어 :  -97.80841064453125

# GridSearchCV
# {'act': 'relu', 'batch_size': 16, 'drop': 0.3, 'node': 128, 'optimizer': 'adam'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001E80D2BEF10>
# -79.78495407104492
# 7/7 [==============================] - 0s 2ms/step - loss: 97.7398 - mae: 6.8532
# 최종 스코어 :  -97.73975372314453

 


