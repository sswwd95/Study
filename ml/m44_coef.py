x = [-3,31,-11,4,0,22,-2,-5,-25,-14]
# y = [-2,32,-10,5,1,23,-1,-4,-24,-13] # weight=1, bias=1
# y = [-5,63,-21,9,1,45,-3,-9,-49,-27] # weight=2, bias=1
y = [-3,65,-19,11,3,47,-1,-7,-47,-25] # weight=2, bias=3



print(x,"\n", y)
# [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14] 
#  [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]
# 데이터의 타입은 list

import matplotlib.pyplot as plt
plt.plot(x,y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X' : x, 'Y' : y}) # 키벨류 형태, 딕셔너리로 지정
print(df)
print(df.shape)
'''
    X   Y
0  -3  -2
1  31  32
2 -11 -10
3   4   5
4   0   1
5  22  23
6  -2  -1
7  -5  -4
8 -25 -24
9 -14 -13
(10, 2)
'''

x_train = df.loc[:,'X'] # 모든행부터 컬럼 x까지
y_train = df.loc[:,'Y']
print(x_train.shape, y_train.shape) #(10,) (10,)
# 스칼라가 10개 (10,1)이면 벡터, (10,3)이면 행렬, (10,4)이면 텐서

print(type(x_train)) # <class 'pandas.core.series.Series'>
# 2차원 이상 들어가면 dataframe

x_train = x_train.values.reshape(len(x_train),1) # (10,1)로 만들어준다는 것
print(x_train.shape, y_train.shape) #(10,1) (10,)

from sklearn.linear_model import LinearRegression # 선형모델의 가장 대표적인 것
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("score : ", score) #score :  1.0

print("기울기 : ",model.coef_) # = weight
print("절편 : ", model.intercept_) # bias
# 기울기 :  [1.]
# 절편 :  1.0

#  weight=2, bias=1
# 기울기 :  [2.]
# 절편 :  1.0

# weight=2, bias=3
# 기울기 :  [2.]
# 절편 :  3.0


