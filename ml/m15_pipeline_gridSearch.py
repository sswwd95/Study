
import numpy as np
from sklearn.datasets import load_iris
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

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) #(150, 4) (150,)

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

# 1번. Pipeline, 변수명 뒤에는 언더바 __ 2개 넣기
parameters = [
    {'a__C' : [1,10,100,1000], 'a__kernel' : ['linear']}, #1,linear / 10,linear/ 100,linear/ 1000,linear => 총 4번
    {'a__C' : [1,10,100], 'a__kernel' : ['rbf'],'a__gamma' : [0.001,0.0001]}, #3*2 =6
    {'a__C' : [1,10,100,1000], 'a__kernel' : ['sigmoid'],'a__gamma' : [0.001,0.0001]}, #4*2 =8 
]

# 2번. make_pipeline
# parameters = [
#     {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['linear']}, #1,linear / 10,linear/ 100,linear/ 1000,linear => 총 4번
#     {'svc__C' : [1,10,100], 'svc__kernel' : ['rbf'],'svc__gamma' : [0.001,0.0001]}, #3*2 =6
#     {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['sigmoid'],'svc__gamma' : [0.001,0.0001]}, #4*2 =8 
# ]


# 2. 모델

# 위에서 전처리하면 과적합. 밑에서 pipeline으로 전처리 해주면 과적합 방지.

# 1번 방법 (이름을 넣어준다)
pipe = Pipeline([('scaler', MinMaxScaler()), ('a',SVC())]) 

# 2번 방법 (이름 없어도 된다.)
# model = make_pipeline(MinMaxScaler(),SVC())
# pipe = make_pipeline(StandardScaler(),SVC())

# model = GridSearchCV(pipe, parameters, cv=5)
model = RandomizedSearchCV(pipe, parameters, cv=5)




model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results) 

# MinMaxScaler
# 0.8333333333333334

# standardscaler
# 0.8666666666666667