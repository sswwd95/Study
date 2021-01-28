# boston = 회귀

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘 중 하나 쓰기
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
# Regressor =회귀모델
# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor  # 단순한 데이터를 대상으로 분류나 회귀를 할 때 사용
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용

dataset = load_boston()
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

# scores : [0.76181326 0.59990688 0.71322067 0.65420905 0.71242571]
# scores : [0.36834491 0.41768535 0.49323364 0.47901334 0.63018265]
# scores : [0.85314854 0.6865445  0.68177562 0.8001429  0.82219906]
# scores : [0.813181   0.8535064  0.85687414 0.88186178 0.89119432]