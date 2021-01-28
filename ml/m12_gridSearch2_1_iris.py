# 모델 RandomForestClassifier

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV # 격자형으로 찾는데 CV까지 하는것
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용

import warnings
warnings.filterwarnings('ignore')

#1. 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) 

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5,shuffle=True)

parameters = [
    {'n_estimatiors' : [100,200]},
    {'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n__jobs' : [-1]},
]

# 2. 모델구성 
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold) 

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print('최적의 매개변수 : ', model.best_estimator_) 
y_pred = model.predict(x_test) 
print('최종정답률 : ', accuracy_score(y_test, y_pred))

# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최종정답률 :  0.9333333333333333

a = model.score(x_test,y_test)
print(a)