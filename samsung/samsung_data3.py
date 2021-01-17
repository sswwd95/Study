import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/csv/samsung.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df1 = df.iloc[:662,[0,1,2,3,5,6]]
df1= df1.sort_index(ascending=True) # 번호 오름차순
y= df1.iloc[:,0:1]
del df1['시가']
df1['시가'] = y
df1 = df1.dropna(axis=0)
print(df1.columns)
print(df1.head())

df2= df.iloc[662:,[0,1,2,3,5,6]] /50
df2= df2.sort_index(ascending=True) # 번호 오름차순
y2= df2.iloc[:,0:1]
del df2['시가']
df2['시가'] = y2
df2 = df2.dropna(axis=0)

df3 = pd.concat([df2,df1])


data = pd.read_csv('./samsung/csv/samsung2.csv',encoding='cp949',thousands=',',index_col=0,header=0)
data = data.iloc[[0],[0,1,2,3,7,8]]
y= data.iloc[:,0:1]
del data['시가']
data['시가'] = y
print(data)
print(data.info())
df3_data = pd.concat([df3,data])


data2 = pd.read_csv('./samsung/csv/samsung3.csv',encoding='cp949',thousands=',',index_col=0,header=0)
data2 = data2.iloc[[0],[0,1,2,3,7,8]]
y= data2.iloc[:,0:1]
del data2['시가']
data2['시가'] = y

f_data = pd.concat([df3_data,data2 ])
print(f_data)  
print(f_data.isnull().sum())

f_df = f_data.to_numpy()
print(f_df)
print(type(f_df)) #<class 'numpy.ndarray'>
print(f_df.shape) #(2397, 6)

np.save('./samsung/npy/samsung3_data.npy', arr = f_df)


