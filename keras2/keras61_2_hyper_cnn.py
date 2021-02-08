# cnn으로 수정
# 파라미터 수정
# 필수  : 노드의 갯수

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D,Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizers='adam', node=64, activation = 'relu',kernel_size=2):
    inputs = Input(shape = (28,28,1),name='input')
    x = Conv2D(128, kernel_size ,padding='same',activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(64, kernel_size, padding = 'same',activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, kernel_size ,padding = 'same', activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(node, activation='relu', name = 'dense1')(x)
    outputs = Dense(10,activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers, metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [32,64,128]
    optimizers = ['adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    node = [128,64,32]
    kernel_size = [2,3]
    activation = ['relu', 'elu', 'prelu', 'softmax']
    return{'batch_size' : batches, 'optimizers' : optimizers, 'drop':dropout, 'node' : node, 
            'activation' : activation, 'kernel_size' : kernel_size}
            # build model에서 정의해준 변수명과 key value 맞춰주기
hyperparameters = create_hyperparameters()
model2 = build_model()

#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)
################ 함수형 모델을 랩핑해야한다. #############################

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)
# search = GridSearchCV(model2,hyperparameters,cv=3)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것
# {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 32}
print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0
# 둘 중 하나만 먹힌다. 

print(search.best_score_) # acc스코어와는 다르다. 
# 0.9572333296140035

acc = search.score(x_test,y_test)
print('최종 스코어 : ',acc)
# 최종 스코어 :  0.9753999710083008

# 랜덤서치 기본값 10번 * cv=3 => 30번돈다.
 


 

