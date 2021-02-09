# 61 카피해서 model.cv_results를 붙여서 완성

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
    x = Dense(64,activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(32,activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [32,64]
    optimizers = ['adam', 'adadelta']
    dropout = [0.2,0.3]
    return{'batch_size' : batches, 'optimizer' : optimizers, 'drop':dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()


modelpath = '../data/modelcheckpoint/k63_cv_results_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.5,verbose=1)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) 
################ 함수형 모델을 랩핑해야한다. #############################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = RandomizedSearchCV(model2,hyperparameters,cv=3)
search = GridSearchCV(model2,hyperparameters,cv=2)

# search.fit(x_train, y_train, verbose=1,eopchs=2,validation_split=0.2) 
search.fit(x_train, y_train, verbose=1,epochs=50,callbacks = [es,lr,cp],validation_split=0.2)
# search.fit(x_train, y_train, verbose=1,epochs=50,callbacks = [es,lr,cp],validation_split=0.2)
# validation split 안쓰려면 es, lr, cp -> loss로 적기


print(search.cv_results_)
# 'RandomizedSearchCV' object has no attribute 'cv_results' 랜덤서치는 오류난다.
print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것
print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것
print(search.best_score_) # acc스코어와는 다르다. 
acc = search.score(x_test,y_test)
print('최종 스코어 : ',acc)
'''
{'mean_fit_time': array([25.57301927, 70.0581131 , 33.49806106, 68.92581415, 14.85608673,
       35.72754741, 15.50792062, 35.59662545]), 'std_fit_time': array([4.25464225, 0.77193165, 3.97762167, 0.48695326, 0.24843073,       
       0.35074401, 0.5278362 , 0.22714818]), 'mean_score_time': array([2.01580894, 1.96576428, 1.93419468, 1.9539938 , 1.09566855,       
       1.12428141, 1.1573925 , 1.10612977]), 'std_score_time': array([0.01546133, 0.00754881, 0.01116073, 0.01214957, 0.00024676,        
       0.04412103, 0.07015824, 0.00139105]), 'param_batch_size': masked_array(data=[32, 32, 32, 32, 64, 64, 64, 64],
             mask=[False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3],
             mask=[False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_optimizer': masked_array(data=['adam', 'adadelta', 'adam', 'adadelta', 'adam',
                   'adadelta', 'adam', 'adadelta'],
             mask=[False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'batch_size': 32, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 32, 'drop': 0.2, 'optimizer': 
'adadelta'}, {'batch_size': 32, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 32, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 64, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 64, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 64, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 64, 'drop': 0.3, 'optimizer': 'adadelta'}], 'split0_test_score': array([0.963     , 0.57743335, 0.95853335, 0.33829999, 0.96160001,
       0.3973    , 0.95796669, 0.39770001]), 'split1_test_score': array([0.96076667, 0.54579997, 0.95466667, 0.38916665, 0.96173334,     
       0.24250001, 0.95616668, 0.44586667]), 'mean_test_score': array([0.96188334, 0.56161666, 0.95660001, 0.36373332, 0.96166667,       
       0.31990001, 0.95706668, 0.42178334]), 'std_test_score': array([1.11666322e-03, 1.58166885e-02, 1.93333626e-03, 2.54333317e-02,    
       6.66677952e-05, 7.73999989e-02, 9.00000334e-04, 2.40833312e-02]), 'rank_test_score': array([1, 5, 4, 7, 2, 8, 3, 6])}
{'batch_size': 32, 'drop': 0.2, 'optimizer': 'adam'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F70BDE7640>
0.9618833363056183
최종 스코어 :  0.9714999794960022
'''

 


 

