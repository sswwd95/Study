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
    x = Dense(512,activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [32,64,128]
    optimizers = ['rnsprop','adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    return{'batch_size' : batches, 'optimizer' : optimizers, 'drop':dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()


modelpath = '../data/modelcheckpoint/k61_mnist2_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.5,verbose=1)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1,epochs=200,validation_split=0.2) # calssifier 여기에 넣어도 된다. 
################ 함수형 모델을 랩핑해야한다. #############################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)
# search = GridSearchCV(model2,hyperparameters,cv=3)

# search.fit(x_train, y_train, verbose=1,eopchs=2,validation_split=0.2) 
search.fit(x_train, y_train, verbose=1,callbacks = [es,lr,cp])

print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 64}

print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0
# 둘 중 하나만 먹힌다. 

print(search.best_score_) # acc스코어와는 다르다. 
# 0.9572333296140035

acc = search.score(x_test,y_test)
print('최종 스코어 : ',acc)
# 최종 스코어 :  0.9822999835014343


# 랜덤서치 기본값 10번 * cv=3 => 30번돈다.
 


 

