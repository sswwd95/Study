import numpy as np
import pandas as pd

ko = pd.read_csv('./samsung/csv/kodex.csv',encoding='cp949', thousands=',',index_col=0, header=0)
ko = ko.iloc[:,[0,1,2,3,7,8]]
ko = ko.sort_index(ascending=True)
y=ko.iloc[:,0:1]
del ko['시가']
ko['시가'] =y
print(ko)
print(ko.isnull())
print(ko.isnull().sum())
ko = ko.dropna(axis=0)
print(ko)

np.save('./samsung/npy/ko_data.npy',arr=ko)


# [1088 rows x 6 columns]