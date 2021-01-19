import numpy as np
import pandas as pd
import os  # 운영체제를 제어할 수 있다.
import glob # 파일들의 리스트를 뽑을 때 사용, 파일의 경로명을 이용해서 마음대로 설정할 수 있다. 
import random # 난수 만들어준다

import warnings
warnings.filterwarnings('ignore') # 경고 메세지 무시

train = pd.read_csv('../solar/train/train.csv')
print(train)
print(train.columns)
print(train.index)
# Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# RangeIndex(start=0, stop=52560, step=1)

train = train.set_index(['Day','Hour','Minute'])
print(train)
print(train.index)
print(type(train))
print(train.isnull().sum()) # 결측치없음

'''
def split_xy(train, time_steps, y_column) : 
    x, y = list(), list()
    for i in range(len(train)) : 
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(train) : 
            break
        tmp_x = train[i:x_end_number, : ]                   
        tmp_y = train[x_end_number:y_end_number, :]         
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy(train, 336, 2) #하루 24시간* (30분씩이니까 )*2 * 7일 = 336 / 다음 2일치 예측해야하니까 2
print(x,'\n', y)
print(x.shape)
print(y.shape)
'''
def preprocess_data(data, is_train = True) : 
    temp = data.copy()
    temp = temp[['DHI','DNI','WS','RH','T','TARGET']]

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

df_train = preprocess_data(train)
df_train.iloc[:48]
print(df_train.iloc[:48])
'''
                DHI  DNI   WS     RH   T     TARGET    Target1    Target2
Day Hour Minute
0   0    0         0    0  1.5  69.08 -12   0.000000   0.000000   0.000000
         30        0    0  1.5  69.06 -12   0.000000   0.000000   0.000000
    1    0         0    0  1.6  71.78 -12   0.000000   0.000000   0.000000
         30        0    0  1.6  71.75 -12   0.000000   0.000000   0.000000
    2    0         0    0  1.6  75.20 -12   0.000000   0.000000   0.000000
         30        0    0  1.5  69.29 -11   0.000000   0.000000   0.000000
    3    0         0    0  1.5  72.56 -11   0.000000   0.000000   0.000000
         30        0    0  1.4  72.55 -11   0.000000   0.000000   0.000000
    4    0         0    0  1.3  74.62 -11   0.000000   0.000000   0.000000
         30        0    0  1.3  74.61 -11   0.000000   0.000000   0.000000
    5    0         0    0  1.3  73.74 -11   0.000000   0.000000   0.000000
         30        0    0  1.3  73.73 -11   0.000000   0.000000   0.000000
    6    0         0    0  1.4  72.22 -12   0.000000   0.000000   0.000000
         30        0    0  1.4  72.22 -12   0.000000   0.000000   0.000000
    7    0         0    0  1.4  70.27 -12   0.000000   0.000000   0.000000
         30        0    0  1.6  64.83 -10   0.000000   0.000000   0.000000
    8    0        29  494  1.8  65.45  -9   7.039287   7.133144   0.750808
         30       61    7  1.9  55.90  -7   5.912871  14.922796   1.689299
    9    0        58  743  2.1  57.39  -6  22.337268  22.806037   5.161690
         30       67  811  1.9  53.15  -4  29.469529   2.158549   4.973992
    10   0       138  368  1.8  55.99  -3  25.339762  36.225679  16.517408
         30      178  224  1.9  55.97  -3  25.152060  41.386914   9.760179
    11   0       193  261  2.0  58.43  -3  28.718397  45.140829   7.601678
         30      189  364  2.3  58.41  -3  33.129393  47.580874  10.980202
    12   0       190   37  2.6  59.19  -3  19.427151  48.612666  17.361857
         30      213  133  2.9  59.18  -3  25.715166  48.143432   4.880090
    13   0       211  114  3.2  63.27  -4  24.589225  46.172141  13.326545
         30      185  101  3.1  63.27  -4  21.304405  42.794162   9.478740
    14   0       124    2  3.0  62.84  -4  11.731500  38.195666   4.504797
         30      135   69  2.7  62.85  -4  14.734764  32.283670   2.627827
    15   0        62    0  2.5  68.55  -5   5.818888  25.432775  14.265504
         30       73   39  2.2  68.55  -5   7.602096  17.737444  16.611987
    16   0        41   11  2.0  70.27  -6   4.035725   9.854244   9.010089
         30       10    0  2.0  70.28  -6   0.938541   2.627827   2.440259
    17   0         0    0  2.0  71.33  -7   0.000000   0.000000   0.000000
         30        0    0  2.0  71.35  -7   0.000000   0.000000   0.000000
    18   0         0    0  2.1  76.43  -8   0.000000   0.000000   0.000000
         30        0    0  2.2  76.44  -8   0.000000   0.000000   0.000000
    19   0         0    0  2.3  76.72  -8   0.000000   0.000000   0.000000
         30        0    0  2.2  76.72  -8   0.000000   0.000000   0.000000
    20   0         0    0  2.2  77.51  -8   0.000000   0.000000   0.000000
         30        0    0  2.0  77.51  -8   0.000000   0.000000   0.000000
    21   0         0    0  1.9  83.46  -9   0.000000   0.000000   0.000000
         30        0    0  1.8  83.46  -9   0.000000   0.000000   0.000000
    22   0         0    0  1.8  83.36  -9   0.000000   0.000000   0.000000
         30        0    0  1.7  83.36  -9   0.000000   0.000000   0.000000
    23   0         0    0  1.7  90.86 -10   0.000000   0.000000   0.000000
         30        0    0  1.6  90.85 -10   0.000000   0.000000   0.000000
'''

train.iloc[48:96]
train.iloc[48+48:96+48]
print(df_train.tail())
'''
Day  Hour Minute
1092 21   30        0    0  3.5  55.97 -1     0.0      0.0      0.0
     22   0         0    0  3.9  54.23 -2     0.0      0.0      0.0
          30        0    0  4.1  54.21 -2     0.0      0.0      0.0
     23   0         0    0  4.3  56.46 -2     0.0      0.0      0.0
          30        0    0  4.1  56.44 -2     0.0      0.0      0.0
'''
print(train.shape) #(52560, 6)

df_test = []  #test 파일 81개 전체를 불러오기
for i in range(81):
    file_path = '../solar/test/' + str(i) +'.csv' #str = 문자열
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test) # test파일 하나로 합치기
print(X_test.shape) #(3888, 6)

from sklearn.model_selection import train_test_split
X1_train, X1_val, Y1_train, Y1_val = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-2], test_size=0.3, random_state=0)
# target1, target2를 빼고 나머지 열에 x1의 train,val/ target1에 y1의 train,val => day7
X2_train, X2_val, Y1_train, Y1_val = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-1], test_size=0.3, random_state=0)
# target1, target2를 빼고 나머지 열에 x2의 train,val/ target2에 y1의 train,val => day8

print(X1_train.head())
'''
                 DHI  DNI   WS     RH   T     TARGET
Day Hour Minute
685 14   0        19    0  2.3  63.64   3   1.783051
211 6    0        45  473  1.8  66.74  19  12.103848
670 11   0        77  929  2.2  39.56  13  56.300682
436 12   30      310  525  1.5  23.71  15  65.401188
751 15   30       70  536  0.5  46.63   0  19.614206
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_val = scaler.transform(X1_val)
X2_train = scaler.transform(X2_train)
X2_val = scaler.transform(X2_val)

print(X1_train.shape) #(36724, 6)
print(X1_val.shape) #(15740, 6)
print(X2_train.shape) #(36724, 6)
print(X2_val.shape) #(15740, 6)

X1_train = X1_train.reshape(X1_train.shape[0],X1_train.shape[1],1)
X1_val =  X1_val.reshape(X1_val.shape[0],X1_val.shape[1],1)
X2_train = X2_train.reshape(X2_train.shape[0],X2_train.shape[1],1)
X2_val =  X2_val.reshape(X2_val.shape[0],X2_val.shape[1],1)

print(X1_train.shape) #(36724, 6,1)
print(X1_val.shape) #(15740, 6,1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(6,1))
dense1 = LSTM(128,activation='relu')(input1)
dense2 = Dense(64)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(64)(dense3)
outputs = Dense(2)(dense4)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../solar/check/solar{eopch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'val_loss', patience=20, mode='min')
hist = model.fit([X1_train,X2_train], [Y1_train,Y2_train], batch_size = 16, callbacks=[es, cp], epochs=1000, validation_data=(x_val,y_val))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])