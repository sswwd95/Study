# randomforest 쓰고 파이프라인 엮어서 25번 돌리기! 데이터는 diabetes


# RandomizedSearch, GS와 pipeline 엮어라.
# 모델은 randomforest

import numpy as np
from sklearn.datasets import load_diabetes
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

dataset = load_diabetes()
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

for train_index, test_index in kfold.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipe = make_pipeline(StandardScaler(),RandomForestRegressor())
    model = RandomizedSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold) 

    print('교차검증 : ', score)

# 교차검증 :  [0.470607   0.372101   0.52796428 0.41738518 0.32867812]
# 교차검증 :  [0.32693694 0.50482199 0.53110705 0.40981443 0.31425523]
# 교차검증 :  [0.51342637 0.41034808 0.4306179  0.35459798 0.50706389]
# 교차검증 :  [0.49142923 0.43733077 0.48308057 0.43521548 0.45395464]
# 교차검증 :  [0.54322705 0.40000956 0.55938155 0.37600591 0.26503874]


# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> : 0.9024578811293587
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'> : 0.8960755941843481
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'> : 0.8990478006656303
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'> : 0.9013484214460155

