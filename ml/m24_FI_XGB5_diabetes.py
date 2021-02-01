# xgboost
# 특징 : 
# 1. gbm보다는 빠르다. 
# 2. 과적합 방지가 가능한 규제가 포함되어 있다. 
# 3. (Classification And Regression Tree)CART를 기반으로 한다. (분류와 회귀 둘 다 가능)



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time

#1. 데이터
dataset = load_diabetes()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)

start = time.time()


#2. 모델
model = XGBRegressor(n_jobs=1) #n_jobs = -1 => cpu 자원을 모두 쓰겠다.


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('기존 acc : ', acc)


sec = time.time()-start
# times = str(datetime.timedelta(seconds=sec)).split(".")
# times = times[0]

print("작업 시간 : ", sec) 

'''
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

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
'''
# Randomforest
# 기존 acc :  0.5150725246657394
# 열 제거 후 acc :  0.4480213168105719

# Gradientboosting
# 기존 acc :  0.4600957097946694
# 열 제거 후 acc :  0.36826891544572193

# xgboost
# 기존 acc :  0.33119109183777795

# n_jobs = -1
# 작업 시간 :  0.11166882514953613
# n_jobs = 8
# 작업 시간 :  0.10970664024353027
# n_jobs = 4
# 작업 시간 :  0.11170315742492676
# n_jobs = 1
# 작업 시간 :  0.14563441276550293

