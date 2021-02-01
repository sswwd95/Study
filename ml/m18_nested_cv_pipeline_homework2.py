# randomforest 쓰고 파이프라인 엮어서 25번 돌리기! 데이터는 wine


# RandomizedSearch, GS와 pipeline 엮어라.
# 모델은 randomforest

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline # 2개가 성능은 똑같다. 사용방법만 조금 다름. 

#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류할 때 사용 =KNN
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier #scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터

dataset = load_wine()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) #(506, 13) (506,)

kfold = KFold(n_splits=5,shuffle=True)

parameters = [
    {'randomforestregressor__n_estimators' : [100,200],
    'randomforestregressor__max_depth' : [6,8,10,12],
    'randomforestregressor__min_samples_leaf' : [3,5,7,10],
    'randomforestregressor__min_samples_split' : [2,3,5,10],
    'randomforestregressor__n_jobs' : [-1]} # n_jobs => cpu를 몇개쓰나? -1이면 다 쓰고, 2는 2개만 쓴다는 것
]

# 2. 모델구성 

# 1번 방법 (이름을 넣어준다)
# pipe = Pipeline([('scaler', MinMaxScaler()), ('a',RandomForestClassifier())]) 

# 2번 방법 (이름 없어도 된다.)
# pipe = make_pipeline(StandardScaler(),RandomForestClassifier())

# model = GridSearchCV(pipe, parameters, cv=5)
# model = RandomizedSearchCV(pipe, parameters, cv=5)
'''
for train_index, test_index in kfold.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipe = make_pipeline(StandardScaler(),RandomForestRegressor())
    model = RandomizedSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold) 

    print('교차검증 : ', score)

# RandomizedSearchCV
# 교차검증 :  [0.93543234 0.85487228 0.96901074 0.89316678 0.94338262]
# 교차검증 :  [0.87426281 0.89829902 0.81738847 0.97577283 0.96424088]
# 교차검증 :  [0.89237401 0.93930709 0.97815776 0.86054145 0.98000671]
# 교차검증 :  [0.8155288  0.89502523 0.89722881 0.95313422 0.93389496]
# 교차검증 :  [0.9264909  0.67960098 0.96456251 0.85736468 0.92135522]]
'''
for train_index, test_index in kfold.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipe = make_pipeline(StandardScaler(),RandomForestRegressor())
    model = GridSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold) 

    print('교차검증 : ', score)


# GridSearchCV
# 교차검증 :  [0.90668466 0.97376456 0.88692238 0.7987271  0.8293004 ]
# 교차검증 :  [0.92787264 0.98683097 0.866575   0.93420667 0.87425479]
# 교차검증 :  [0.89458781 0.89850472 0.88432365 0.9430782  0.90728992]
# 교차검증 :  [0.95486858 0.63078286 0.83375931 0.93487804 0.89970853]
# 교차검증 :  [0.9676283  0.79626842 0.96706261 0.95521676 0.93165948]

