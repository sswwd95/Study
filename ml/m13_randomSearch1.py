# 파라미터를 랜덤으로 돌린다. 속도가 그리드서치보다 빠르다는 장점이 있다.
# 모든 파라미터가 중요하지 않을 때 사용. 최적의 파라미터 있으면 그리드가 나음

# n_iter int, default=10  

# iris = 다중분류

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
from sklearn.metrics import accuracy_score

#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류할 때 사용 =KNN
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier #scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import datetime
import time
start = time.time()

#1. 데이터

# dataset = load_iris()
# x = dataset.data
# y = dataset.target

# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
print(x.shape,y.shape) 

data = dataset.to_numpy()
print(data)
print(type(data))
#<class 'numpy.ndarray'>

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5,shuffle=True) # train만 5조각으로 나눈다


parameters = [
    {'C' : [1,10,100,1000], 'kernel' : ['linear']}, #1,linear / 10,linear/ 100,linear/ 1000,linear => 총 4번
    {'C' : [1,10,100], 'kernel' : ['rbf'],'gamma' : [0.001,0.0001]}, #3*2 =6
    {'C' : [1,10,100,1000], 'kernel' : ['sigmoid'],'gamma' : [0.001,0.0001]}, #4*2 =8 
]
#key value 쌍으로 이루어진 딕셔너리들

# 2. 모델구성 
# model = SVC()
model = RandomizedSearchCV(SVC(), parameters, cv=kfold) 

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print('최적의 매개변수 : ', model.best_estimator_) # 90번을 돌 동안 가장 좋은 값을 출력해준다. 
y_pred = model.predict(x_test) # 가장 좋은 값을 빼줘서 넣어준다. 
print('최종정답률 : ', accuracy_score(y_test, y_pred))

# GridsearchCV
# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최종정답률 :  0.9333333333333333

# RandomizedSearchCV
# 최적의 매개변수 :  SVC(C=100, kernel='linear')
# 최종정답률 :  0.9333333333333333

a = model.score(x_test,y_test)
print(a)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("작업 시간 : ", times) #작업 시간 :  0:00:00

np.save('../data/npy/iris_sklearn.npy', arr=data)
