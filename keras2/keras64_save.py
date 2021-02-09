# 가중치 저장할 것
# 1. model.save()
# 2. pickle 쓸 것

# 61 카피해서 model.cv_results를 붙여서 완성

######### pickle 저장 #############
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape = (28*28,),name='input')
    x = Dense(128,activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64,activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(32,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [32,64]
    optimizers = ['adam', 'adadelta']
    dropout = [0.1,0.2]
    return{'batch_size' : batches, 'optimizer' : optimizers, 'drop':dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()


modelpath = '../data/modelcheckpoint/k61_mnist2_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.5,verbose=1)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1,epochs=50,validation_split=0.2) # calssifier 여기에 넣어도 된다. 
################ 함수형 모델을 랩핑해야한다. #############################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2,hyperparameters,cv=3)
search = GridSearchCV(model2,hyperparameters,cv=2)

import pickle
# pickle.dump(model2, open('../data/pickle/keras64.pickle.data', 'wb')) # 저장

model4 = pickle.load(open('../data/pickle/keras64.pickle.data', 'rb')) # 불러오기

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train,y_train, verbose=1)

search.best_estimator_.model.save('../data/h5/k64_modelbest.h5')

print(search.best_params_)

print(search.best_score_)

acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)
