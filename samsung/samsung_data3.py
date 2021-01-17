import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/csv/samsung.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df = df.iloc[:662,[0,1,2,3,5,6]]
df= df.sort_index(ascending=True) # 번호 오름차순
y= df.iloc[:,0:1]
del df['시가']
df['시가'] = y
df1 = df.dropna(axis=0)
print(df1.columns)
print(df1.head())
print(df)
print(df.shape)

df2 = pd.read_csv('./samsung/csv/samsung2.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df2 = df2.iloc[[0],[0,1,2,3,7,8]]
y= df2.iloc[:,0:1]
del df2['시가']
df2['시가'] = y
print(df2)
print(df2.info())
df3 = pd.concat([df,df2])
print(df3)

df4 = pd.read_csv('./samsung/csv/samsung3.csv',encoding='cp949',thousands=',',index_col=0,header=0)
df4 = df4.iloc[[0],[0,1,2,3,7,8]]
y= df4.iloc[:,0:1]
del df4['시가']
df4['시가'] = y

f_data = pd.concat([df3,df4])
print(f_data)  
print(f_data.isnull().sum())

f_df = f_data.to_numpy()
print(f_df)
print(type(f_df)) #<class 'numpy.ndarray'>
print(f_df.shape) #(664, 6)

np.save('./samsung/npy/samsung3_data.npy', arr = f_df)


