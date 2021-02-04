# xgboost
# plot_importance 제공된다. 대신 f score 기준으로 나온다. 

# f score?
# f1 score이라고 하며, dataset에서 모델의 정확도를 측정한 것이다. 
# '양성'또는 '음성'으로 분류하는 이진 분류 시스템을 평가하는데 사용된다. 
# 모델의 정밀도와 재현율을 결합하는 방법이며, 이 둘의 조화 평균으로 정의된다. 



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time



#1. 데이터
dataset = load_iris()

x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)


start = time.time()

#2. 모델
model = XGBClassifier(n_jobs=8) #n_jobs = -1 => cpu 자원을 모두 쓰겠다.


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

sec = time.time()-start
# times = str(datetime.timedelta(seconds=sec)).split(".")
# times = times[0]

print("작업 시간 : ", sec) 
print(model.feature_importances_)
print('기존 acc : ', acc)

####################### 중요도 낮은 피처 number만큼 반환 ###########################
def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result
print(cut_columns(model.feature_importances_,dataset.feature_names,2))
##################################################################################
'''
def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
'''
plot_importance(model)
plt.show()


# Randomforest
# 기존 acc :  0.8666666666666667
# acc :  0.9666666666666667

# Gradientboosting
# 기존 acc :  0.8666666666666667
# 열 제거 후 acc :  0.9666666666666667

# xgboost
# 기존 acc :  0.9333333333333333

# n_jobs = -1
# 작업 시간 :  0.06080961227416992
# n_jobs = 8
# 작업 시간 :  0.058841705322265625
# n_jobs = 4
# 작업 시간 :  0.05880117416381836
# n_jobs = 1
# 작업 시간 :  0.05709576606750488