# diabetes = 회귀

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘 중 하나 쓰기
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# Regressor =회귀모델
# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor  # 단순한 데이터를 대상으로 분류나 회귀를 할 때 사용
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용

dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape) #(178,13)
print(y.shape) #(178,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 55
)

x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                test_size = 0.2, shuffle = True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 2. 모델구성 -> 통상적으로 밑으로 갈 수록 좋다고 하는 모델.
# model = LinearRegression()
# model = KNeighborsRegressor()
model = DecisionTreeRegressor()
# model = RandomForestRegressor() # ==> rf = RandomForestClassifier(), 'model'은 변수명 정의한 것. 

# 3. 훈련

model.fit(x_train,y_train)

#4. 평가, 예측

#result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test) # evaluate에서 나오는건 loss,acc였는데 score하면 acc바로 나옴. == model.score는 evaluate의 metrics에서 acc값과 같다. 
print('result : ', result)

y_predict = model.predict(x_test) # predict는 머신러닝과 동일하게 사용
# print(y_predict)

r2 = r2_score(y_test, y_predict)
print('r2_score : ',r2)
R2 :  0.5094325786699055


############################
# Tensorflow
# R2 : 0.5094325786699055
############################

# LinearRegression() 
# result :  0.5122371543531412
# r2_score :  0.5122371543531412

# KNeighborsRegressor()
# result :  0.5529025892531771
# r2_score :  0.5529025892531771

# DecisionTreeRegressor()
# result :  -0.05075229477185794
# r2_score :  -0.05075229477185794

# RandomForestRegressor()
# result :  0.4630611282331777
# r2_score :  0.4630611282331777

# DecisionTreeRegressor 모델이 제일 값 떨어진다. 다른건 돌릴때마다 달라져서 비슷..?


'''
머신러닝 기본값으로 해도 딥러닝을 이기는 경우가 많다. 딥러닝 할 때 로스 지표를 머신러닝으로 잡고 이기려고 해보기
'''
