# gridsearch = 격자로 촘촘하게 다 찾겠다.
# 케라스에 대한 파라미터 값 다 넣고 돌려볼 것이다.
# iris = 다중분류

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV # 격자형으로 찾는데 CV까지 하는것
from sklearn.metrics import accuracy_score

#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류할 때 사용 =KNN
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier #scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용

import warnings
warnings.filterwarnings('ignore')

#1. 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) 
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1
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
model = GridSearchCV(SVC(), parameters, cv=kfold) 
#SVC모델을 돌릴건데 GridsearchCV 식으로 5번 돌린다.
# 모델 뒤에 딕셔너리 형태의 파라미터 추가. 
#(4+6+8)*5 = 90 총 90번 돌리는 것

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print('최적의 매개변수 : ', model.best_estimator_) # 90번을 돌 동안 가장 좋은 값을 출력해준다. 
y_pred = model.predict(x_test) # 가장 좋은 값을 빼줘서 넣어준다. 
print('최종정답률 : ', accuracy_score(y_test, y_pred))

# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최종정답률 :  0.9333333333333333

a = model.score(x_test,y_test)
print(a)