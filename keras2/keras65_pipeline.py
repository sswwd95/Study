# 61번을 파이프라인으로 구성

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터/ 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


#2. 모델 구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name = 'input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, verbose=1, batch_size=32, epochs=1)
'''
################################# 파이프라인 정의 #######################################
#pipeline(파라미터에 설정한 이름 써주기)
pipeline = Pipeline([('scaler', MinMaxScaler()), ('clf', model2)])

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'clf__batch_size' : batches, 'clf__optimizer' : optimizers, 'clf__drop' : dropouts}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(pipeline, hyperparameters, cv = 3)

search.fit(x_train, y_train)

print('best_params_: ', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)

# best_params_:  {'clf__optimizer': 'adam', 'clf__drop': 0.2, 'clf__batch_size': 40}
# 250/250 [==============================] - 0s 1ms/step - loss: 0.0990 - acc: 0.9681
# 최종 스코어:  0.9681000113487244
'''
#########################################################################################
# make pipeline (파라미터에 모델명 붙여서 써주기)
pipeline = make_pipeline(MinMaxScaler(),model2)

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'KerasClassifier__batch_size' : batches, 'KerasClassifier__optimizer' : optimizers, 'KerasClassifier__drop' : dropouts}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(pipeline, hyperparameters, cv = 3)

search.fit(x_train, y_train)

print('best_params_: ', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)