
# train, test 나눈 다음에 train만 val 하지말고, 
# kfold 한 후에 train_test_split 사용

# test 부분을 사용 안했다. 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 값
from sklearn.metrics import accuracy_score
#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류할 때 사용 =KNN
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier #scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용


#1. 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape,y.shape) 
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

# 머신러닝은 원핫인코딩 필요없음.

kfold = KFold(n_splits=5,shuffle=True) # train만 5조각으로 나눈다

models = [LinearSVC, SVC, KNeighborsClassifier,DecisionTreeClassifier, RandomForestClassifier]

'''
#train_idx와 val_idx를 생성
for train_idx, val_idx in kfold.split(x): 
    # train fold, val fold 분할
    x_train = x[train_idx]
    x_test = x[train_idx]
    y_train = y[train_idx]
    y_test = y[train_idx]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model,x_train, y_train, cv=kfold) 
    print('scores :', scores) 
'''

for train_idx, val_idx in kfold.split(x): 
    # train fold, val fold 분할
    x_train = x[train_idx]
    x_test = x[train_idx]
    y_train = y[train_idx]
    y_test = y[train_idx]
    print(train_idx)

# kf = KFold(n_splits=2)
# for train, test in kf.split(x):

# # scores : [0.83333333 1.         0.95833333 0.91666667 1.        ]
# # scores : [0.95833333 0.83333333 0.91666667 0.95833333 1.        ]
# # scores : [0.95833333 0.83333333 0.91666667 0.91666667 1.        ]
# # scores : [0.875      0.95833333 0.95833333 0.91666667 0.95833333]
# # scores : [0.91666667 0.91666667 0.95833333 0.95833333 1.        ]


'''
x_train,x_val,y_train,y_val = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)


# 2. 모델구성 
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()  

# fit과 scores 다 돌려진 것
# scores : [0.93333333 0.86666667 0.96666667 1.         0.93333333] -> x,y를 kfold 했을 때
# scores : [0.95833333 1.         0.95833333 1.         0.91666667] -> x_train, y_train을 kfold 했을 때
'''
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
# scores : [0.875 1.    1.    1.    1.   ]