import numpy as np
import pandas as pd

df = pd.read_csv('./solar/csv/train.csv',index_col=None, header=0)
print(df)
print(df.columns)
print(df.index)
# Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# RangeIndex(start=0, stop=52560, step=1)

df = df.set_index(['Day','Hour','Minute'])
print(df)
print(df.index)
print(type(df))
print(df.isnull().sum()) # 결측치없음
df = df.to_numpy()

def split_xy(df, time_steps, y_column) : 
    x, y = list(), list()
    for i in range(len(df)) : 
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(df) : 
            break
        tmp_x = df[i:x_end_number, : ]                   
        tmp_y = df[x_end_number:y_end_number, :]         
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy(df, 336, 2) #하루 24시간* (30분씩이니까 )*2 * 7일 = 336 / 다음 2일치 예측해야하니까 2
print(x,'\n', y)
print(x.shape)
print(y.shape)

