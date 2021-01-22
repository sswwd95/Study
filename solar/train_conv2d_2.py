# 'hour'넣기


import numpy as np
import pandas as pd
import os  # 운영체제를 제어할 수 있다.
import glob # 파일들의 리스트를 뽑을 때 사용, 파일의 경로명을 이용해서 마음대로 설정할 수 있다. 
import random # 난수 만들어준다
import tensorflow.keras.backend as K


import warnings
warnings.filterwarnings('ignore') # 경고 메세지 무시
train = pd.read_csv('../solar/train/train.csv',index_col=None, header=0)
sub = pd.read_csv('./solar/csv/sample_submission.csv')

print(train)
print(train.shape) #(52560, 9)
print(train.columns)
print(train.index)
# Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# RangeIndex(start=0, stop=52560, step=1)
print(train)
print(train.index)
print(type(train))
print(train.isnull().sum()) # 결측치없음

# 추가해보기!
# def Add_features(data):
#      c = 243.12
#      b = 17.62
#      gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
#      dp = ( c * gamma) / (b - gamma)
#      data.insert(1,'Td',dp)
#      data.insert(1,'T-Td',data['T']-data['Td'])
#      data.insert(1,'GHI',data['DNI']+data['DHI'])
#      return data

def preprocess_data(data, is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12-6)/6*np.pi/2) 
    data.insert(1, 'GHI', data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # day7
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # day8
        temp = temp.dropna()
        return temp.iloc[:-96] # day8에서 2일치 땡겨서 올라갔기 때문에 마지막 2일 빼주기

    elif is_train==False:
        temp = temp[['Hour','TARGET','GHI','DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:,:] # 트레인데이터가 아니면 마지막 하루만 리턴시킴

df_train = preprocess_data(train)
train = df_train.to_numpy()

print(train)
print(train.shape) #(52464, 10) day7,8일 추가해서 컬럼 10개

###### test파일 합치기############
df_test = []

for i in range(81):
    file_path = '../solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) # 위에서 명시한 False => 마지막 하루만 리턴
    df_test.append(temp)   # 마지막 하루 값들만 전부 붙여주기 

x_test = pd.concat(df_test)
print(x_test.shape) #(3888, 8) -> (81, 48,8) 81일, 하루(24*2(30분단위)=48), 8개 컬럼
target = x_test.to_numpy()

print(target.shape) #(3888, 8)

#####################시계열 데이터 자르기######################
def split_xy(train, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(train)):
        if i > len(train)-x_row:
            break
        tmp_x = train[i:i+x_row, :x_col]
        tmp_y = train[i:i+x_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y= split_xy(train, 48, 8, 48,2) 

print(x.shape) #(52417, 48, 8)
print(y.shape) #(52417, 48, 2)

################### 2차원 만들기 #########################
target = target.reshape(81, 48, 8)
target = target.reshape(target.shape[0], target.shape[1]*target.shape[2])
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

################## train / test 분리 ######################

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 0)

print(x_train)
print(y_train)
print(x_train.shape) #(41933, 48, 7)
print(x_test.shape) #(10484, 48, 7)
print(y_train.shape) #(41933, 48, 2)
print(y_test.shape) #(10484, 48, 2)

###################정규화 ################# 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
target = scaler.transform(target)

##############################################

# 4차원으로 만들어주기
x_train = x_train.reshape(x_train.shape[0], 1, 48, 8)
x_test = x_test.reshape(x_test.shape[0], 1, 48, 8)
target = target.reshape(target.shape[0], 1, 48, 8)

y_train = y_train.reshape(y_train.shape[0], 1, 48, 2)
y_test = y_test.reshape(y_test.shape[0], 1, 48, 2)

################ 퀀타일 로스 ####################
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#################################################

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten,Reshape
from tensorflow.keras.backend import mean, maximum

def Model():
    model = Sequential()
    model.add(Conv2D(128,2,padding='same',activation='relu', input_shape = (1,48,8)))
    model.add(Dropout(0.2))
    model.add(Conv2D(118,2,padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(108,2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(98, activation='relu'))
    model.add(Dense(96))
    model.add(Reshape((48,2)))
    model.add(Dense(2, activation='relu'))
    return model

for q in quantiles:
    # 컴파일, 훈련
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    model = Model()
    modelpath = '../solar/check/solar0122_{epoch:02d}_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=16, validation_split=0.2, callbacks=[es,lr])


    #평가, 예측
    loss = model.evaluate(x_test, y_test, batch_size=16)
    print('loss: ', loss)
  
    y_pred = model.predict(target)
   
    print(y_pred.shape)

    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]))
    df_y_pred = pd.concat([y_pred], axis=1)
    df_y_pred[df_y_pred<0] = 0
    df_y_pred = df_y_pred.to_numpy()
        
    print(str(q)+'번째 지정')
    sub.loc[sub.id.str.contains('Day7'), 'q_' + str(q)] = df_y_pred[:,0].round(2)
    sub.loc[sub.id.str.contains('Day8'), 'q_' + str(q)] = df_y_pred[:,1].round(2)


sub.to_csv('./solar/csv/sub_conv2d_2.csv',index=False)

