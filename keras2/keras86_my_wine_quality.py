# 실습. 만들기!
import warnings
warnings.filterwarnings('ignore')

import numpy as pd
import pandas as pd

df = pd.read_csv('../data/csv/winequality-white.csv',sep=';')
print(df)

wine=df.to_numpy()
print(wine)

x = wine[:,:-1]
y = wine[:,-1]
print(x)
print(y) 

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape) # (4898, 10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (3918, 11) (980, 11)
# (3918, 10) (980, 10)

from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV # 격자형으로 찾는데 CV까지 하는것
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline 
import matplotlib.pyplot as plt
############################################################################
'''
parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1]}
]

# 2. 모델구성 
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5) 

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print('최적의 매개변수 : ', model.best_estimator_) 
y_pred = model.predict(x_test) 
print('최종정답률 : ', accuracy_score(y_test, y_pred))

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3)
# 최종정답률 :  0.5642857142857143
'''
###############################################################################
'''
parameters = [
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bytree' : [0.6,0.8,1]},
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bylevel' : [0.6,0.8,1]}
]

pipe = Pipeline([('scaler', MinMaxScaler()),('a', XGBClassifier(n_jobs=-1))])

# 2. 모델구성
model = RandomizedSearchCV(pipe, parameters, cv=5)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

# scaler는 powertransformer이 젤 잘 나온다
# RobustScaler
#  acc: 0.6806122448979591

# MaxAbsScaler
# acc: 0.6714285714285714

# PowerTransformer
# acc: 0.6816326530612244

# StandardScaler
# acc: 0.6806122448979591

# MinMaxScaler
# acc: 0.6704081632653062
'''
###################################################################################


#--------------------- y 조절하기(임의로 가능) -----------------------
newlist = []
for i in list(y):
    if i <= 4: # i는 0번째부터 들어간다. (3,4) => 0등급
        newlist +=[0]
    elif i <= 7:  #(5,6,7) => 1등급
        newlist +=[1]
    else :        # (8, 9) => 2등급 표본은 임의로 잡는다
        newlist +=[2]
y = newlist
#--------------------------------------------------------------------

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_test = hot.transform(y_test).toarray()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# (3918, 11) (980, 11)
# (3918, 7) (980, 7)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape = (11,),name='input')
    x = Dense(512,activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(7,activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [32,64,128]
    optimizers = ['adam', 'adadelta'] # Unknown optimizer: rnsprop
    dropout = [0.1,0.2,0.3]
    return{'batch_size' : batches, 'optimizer' : optimizers, 'drop':dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k86_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.5,verbose=1)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


#######################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1,epochs=100,validation_split=0.2) # calssifier 여기에 넣어도 된다. 
################ 함수형 모델을 랩핑해야한다. #############################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)
# search = GridSearchCV(model2,hyperparameters,cv=3)

# search.fit(x_train, y_train, verbose=1,eopchs=2,validation_split=0.2) 
search.fit(x_train, y_train, verbose=1,callbacks = [es,lr,cp])

print(search.best_params_) # 내가 선택한 파라미터 중 제일 좋은 것
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 64}

# print(search.best_estimator_) # 전체 추정기 중에서 가장 좋은 것
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0
# 둘 중 하나만 먹힌다. 

print(search.best_score_) # acc스코어와는 다르다. 
# 0.9572333296140035

acc = search.score(x_test,y_test)
print('최종 스코어 : ',acc)

# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 128}
# 0.5632975896199545
# 8/8 [==============================] - 0s 737us/step - loss: 1.0083 - acc: 0.5949
# 최종 스코어 :  0.594897985458374


# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 64}
# 0.5684022506078085
# 16/16 [==============================] - 0s 2ms/step - loss: 1.0234 - acc: 0.5867
# 최종 스코어 :  0.5867347121238708