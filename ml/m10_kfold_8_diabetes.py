# diabetes = 회귀

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘 중 하나 쓰기
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5,shuffle=True) # train만 5조각으로 나눈다

# 2. 모델구성

models = [ LinearRegression, KNeighborsRegressor,DecisionTreeRegressor, RandomForestRegressor]
for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model,x_train, y_train, cv=kfold) 
    print('scores :', scores) 

'''
# 3. 훈련
model.fit(x,y)

#4. 평가, 예측

result = model.score(x_test,y_test) 
print('result : ', result)

y_predict = model.predict(x_test) 
# print(y_predict)
print(np.argmax(y_predict,axis=-1))

acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ',acc)

# result = model.score(x_test,y_test) == acc = accuracy_score(y_test, y_predict) 값이 동일함.
'''
# models = [ LinearRegression, KNeighborsRegressor,DecisionTreeRegressor, RandomForestRegressor]
# RandomForestRegressor 값 제일 좋음

# scores : [0.47164056 0.45778341 0.36953392 0.58664636 0.53566347]
# scores : [0.44782144 0.32027122 0.30163415 0.28521314 0.37419195]
# scores : [-0.01246015 -0.0720414  -0.07891727 -0.19838204 -0.05551894]
# scores : [0.50875553 0.49298898 0.4000794  0.40859242 0.38146359]