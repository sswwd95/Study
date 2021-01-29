# RandomizedSearch, GS와 pipeline 엮어라.
# 모델은 randomforest

# 끝판왕?!
# 전처리까지 합치는 걸 pipeline
import numpy as np
from sklearn.datasets import load_breast_cancer
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
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) #(569, 30) (569,)

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

parameters = [
    {'a__n_estimators' : [100,200],
    'a__max_depth' : [6,8,10,12],
    'a__min_samples_leaf' : [3,5,7,10],
    'a__min_samples_split' : [2,3,5,10],
    'a__n_jobs' : [-1]} # n_jobs => cpu를 몇개쓰나? -1이면 다 쓰고, 2는 2개만 쓴다는 것
]

# 2. 모델구성 

# 1번 방법 (이름을 넣어준다)
pipe = Pipeline([('scaler', MinMaxScaler()), ('a',RandomForestClassifier())]) 

# 2번 방법 (이름 없어도 된다.)
# pipe = make_pipeline(StandardScaler(),RandomForestClassifier())

# model = GridSearchCV(pipe, parameters, cv=5)
model = RandomizedSearchCV(pipe, parameters, cv=5)


model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results) 

# MinMaxScaler
# 0.8333333333333334

# standardscaler
# 0.8666666666666667

# grid
# 0.9298245614035088

# random
# 0.9473684210526315

