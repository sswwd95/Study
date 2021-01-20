import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./solar/csv/train.csv')
print(train.tail())
'''
        Day  Hour  Minute  DHI  DNI   WS     RH  T  TARGET
52555  1094    21      30    0    0  2.4  70.70 -4     0.0     
52556  1094    22       0    0    0  2.4  66.79 -4     0.0     
52557  1094    22      30    0    0  2.2  66.78 -4     0.0     
52558  1094    23       0    0    0  2.1  67.72 -4     0.0     
52559  1094    23      30    0    0  2.1  67.70 -4     0.0
'''
sub = pd.read_csv('./solar/csv/sample_submission.csv')

def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['TARGET','DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]
    elif is_train==False:
        temp = temp[['TARGET','DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:,:]
df_train = preprocess_data(train)

###### test파일 합치기############
df_test = []

for i in range(81):
    file_path = '../solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
print(X_test.shape) #(3888, 6)
print(X_test.head(48))

####################################

from sklearn.model_selection import train_test_split
x1_train, x1_val, y1_train, y1_val = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-2], test_size=0.2, random_state=0)
x2_train, x2_val, y2_train, y2_val = train_test_split(
    df_train.iloc[:,:-2],df_train.iloc[:,-1], test_size=0.2, random_state=0)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

