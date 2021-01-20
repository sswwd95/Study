import numpy as np
import pandas as pd
import os  # 운영체제를 제어할 수 있다.
import glob # 파일들의 리스트를 뽑을 때 사용, 파일의 경로명을 이용해서 마음대로 설정할 수 있다. 
import random # 난수 만들어준다

import warnings
warnings.filterwarnings('ignore') # 경고 메세지 무시
train = pd.read_csv('../solar/train/train.csv')
print(train)
print(train.shape) #(52560, 9)
print(train.columns)
print(train.index)
# Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# RangeIndex(start=0, stop=52560, step=1)

train = train.set_index(['Day','Hour','Minute'])
print(train)
print(train.index)
print(type(train))
print(train.isnull().sum()) # 결측치없음

def preprocess_data(data, is_train = True) : 
    temp = data.copy()
    temp = temp[['DHI','DNI','WS','RH','T','TARGET']]
    # print(is_train)        

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        # shift는 값을 땡기는 것. (-48)이면 위로 48만큼 올린다. (태양광 데이터에선 48은 하루치 )
        # fillna는 결측값을 설정. ffill 또는 pad = 결측값을 앞 방향으로 채워나감/ bfill 또는 backfill은 뒷방향으로 채워나감
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        # target1 = day7, target2 = day8
        temp = temp.dropna() # 결측치 있는 행 삭제
        return temp.iloc[:-96] # 뒤에서부터 2일치 데이터 리턴 

    elif is_train==False: 
        temp = temp[['DHI', 'DNI', 'WS','RH','T','TARGET']]  # target1, target2 없이 열의 자료만 가져오는 것
        return temp.iloc[-48:, :]

train = preprocess_data(train)

print(type(train))
print(train.shape) #(52464, 8)

train= train.to_numpy()

def split_xy(train,x_row,x_col,y_row,y_col):
    x,y = [],[]
    for i in range(len(train)):
        x_end = i + x_row    
        y_end = x_end   
        if y_end>len(train):
            break
        x_tmp = train[i:x_end,:x_col]   
        y_tmp = train[x_end-1:y_end,-2:]
        x.append(x_tmp)
        y.append(y_tmp) 
    return np.array(x), np.array(y)
x, y = split_xy(train, 48, 6, 1, 2) 

print(x, "\n", y)
print(x)
print(y)
print("x.shape : ", x.shape) #(52417, 48, 6)
print("y.shape : ", y.shape) #(52417, 1, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 0)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle = True, random_state = 0)

print(x_train)
print(y_train)
print(x_train.shape) #(33546, 48, 6)
print(y_train.shape) #(33546, 1, 2)

df_test = []  #test 파일 81개 전체를 불러오기
for i in range(81):
    file_path = '../solar/test/' + str(i) +'.csv' #str = 문자열
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

df_test = pd.concat(df_test) # test파일 하나로 합치기

print(df_test)
print(df_test.shape) #(3888, 6)

X_test = df_test.to_numpy()
x_pred=X_test.reshape(81,48,6)

print(x_pred.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D

model = Sequential()
model.add(LSTM(64,activation='relu', input_shape=(48,6)))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2))

model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
modelpath = '../solar/check/solar_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, batch_size = 16, callbacks=[es, cp, lr], epochs=20, validation_data=(x_val,y_val))


#평가, 예측
loss = model.evaluate(x_test,y_test, batch_size=16)
print('loss : ',loss)

y_pred = y_pred.reshape(3888, 6)


y_pred = model.predict(x_pred)
print(y_pred)

y_pred.to_csv('./solar/csv/submission_2.csv', sep=',')
