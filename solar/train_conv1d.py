# 'hour'빼고 돌려보기


import numpy as np
import pandas as pd
import os  # 운영체제를 제어할 수 있다.
import glob # 파일들의 리스트를 뽑을 때 사용, 파일의 경로명을 이용해서 마음대로 설정할 수 있다. 
import random # 난수 만들어준다
import tensorflow.keras.backend as K


import warnings
warnings.filterwarnings('ignore') # 경고 메세지 무시
train = pd.read_csv('../solar/train/train.csv')
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

######### test파일 하나로 합치기 #####
def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:, :]   #테스트 파일의 day6부분만 자르기

df_test = []
for i in range(81):
      file_path = '../solar/test/' + str(i) + '.csv'
      temp = pd.read_csv(file_path)
      temp = preprocess_data(temp)  #day6만 모아서 붙이기
      df_test.append(temp)

X_test = pd.concat(df_test)
print(X_test.shape) #(3888, 9)

# 직관적인 GHI 피처 추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# def Add_features(data):
#      c = 243.12
#      b = 17.62
#      gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
#      dp = ( c * gamma) / (b - gamma)
#      data.insert(1,'Td',dp)
#      data.insert(1,'T-Td',data['T']-data['Td'])
#      data.insert(1,'GHI',data['DNI']+data['DHI'])
#      return data

train = Add_features(train)
X_test = Add_features(X_test)

print(train.shape) #(52560, 10)
print(X_test.shape) #(3888, 10)

train = train.drop(['Day','Hour','Minute'],axis=1)  
X_test  = X_test.drop(['Day','Hour','Minute'],axis=1)

print(train.columns) #Index(['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')

temp = train.copy()
temp = temp[['GHI','DHI', 'DNI', 'WS', 'RH', 'T','TARGET']]

temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
temp = temp.dropna()
train = temp.iloc[:-96]

print(train.columns) #Index(['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET', 'Target1', 'Target2'], dtype='object')
print(train.tail())
print(train.shape) #(52464, 9)
print(X_test.columns)
print(X_test.shape) #(3888, 7)

# print('===================')
# train = train.to_numpy()
X_test = X_test.to_numpy()
train = train.values

X_test = X_test.reshape(81, 48, 7)

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

x, y = split_xy(train, 48, 7, 48,2) 

print(x.shape) #(52417, 48, 7)
print(y.shape) #(52417, 48, 2)

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

##################### 정규화 ################# 차원 상관없이 되나,,?
num_features = x_train.shape[2] # 정규화는 피처만 하는 것

train_mean = x_train.mean() # mean() = 평균계산
train_std  = x_train.std()  # std() = 표준 편차 계산

x_train= (x_train - train_mean) / train_std
x_test  = (x_test - train_mean) / train_std

##############################################

# 4차원으로 만들어주기
x_train = x_train.reshape(x_train.shape[0], 1, 48, 7)
x_test = x_test.reshape(x_test.shape[0], 1, 48, 7)
X_test = X_test.reshape(X_test.shape[0], 1, 48, 7)

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
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.backend import mean, maximum

def Model():
    model = Sequential()
    model.add(Conv2D(128,2,padding='same',activation='relu', input_shape = (1,48,7)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,2,padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    return model

for q in quantiles:
    # 컴파일, 훈련
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    model = Model()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam')
    es = EarlyStopping(monitor ='val_loss', patience=10, mode='min')
    rl = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)
    filepath = '../solar/check/solar_conv2d_{epoch:02d}_{val_loss:.4f}.hdf5'
    check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', save_best_only=True, mode='min') 
    hist = model.fit(x_train, y_train, epochs=1, batch_size=16, validation_split=0.2, callbacks=[es,rl])


    #평가, 예측
    result = model.evaluate(x_test, y_test, batch_size=16)
    print('loss: ', result[0])
    print('mae: ', result[1])
    y_pred = model.predict(X_test)
   
    print(y_pred.shape)

    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2]))
    df_y_pred = pd.concat([y_pred], axis=1)
    df_y_pred[df_y_pred<0] = 0
    df_y_pred = df_y_pred.to_numpy()
        
    print(str(q)+'번째 지정')
    subfile.loc[subfile.id.str.contains('Day7'), 'q_' + str(q)] = df_y_pred[:,0].round(2)
    subfile.loc[subfile.id.str.contains('Day8'), 'q_' + str(q)] = df_y_pred[:,1].round(2)


sub.to_csv('./solar/csv/sub_conv2d.csv',index=False)

