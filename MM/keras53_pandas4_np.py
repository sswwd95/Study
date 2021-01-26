import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

print(df) 
print(df.shape) #(150, 5)
print(df.info())

'''
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   Target        150 non-null    int64 
dtypes: float64(4), int64(1)
memory usage: 7.0 KB
None
'''
#pandas를 numpy로 바꾸기
aaa= df.to_numpy()
print(aaa)
print(type(aaa)) #<class 'numpy.ndarray'>

bbb = df.values
print(bbb)
print(type(bbb))

np.save('../data/npy/iris_sklearn.npy', arr=aaa)
# np.save('../data/npy/iris_sklearn.npy', arr=bbb) 두 개 같음. 

# 과제 : 판다스의 loc iloc에 대해 정리