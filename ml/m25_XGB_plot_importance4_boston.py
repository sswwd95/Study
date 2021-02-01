# xgboost
# 특징 : 
# 1. gbm보다는 빠르다. 
# 2. 과적합 방지가 가능한 규제가 포함되어 있다. 
# 3. (Classification And Regression Tree)CART를 기반으로 한다. (분류와 회귀 둘 다 가능)



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor,plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time

#1. 데이터
dataset = load_boston()
x_train,x_test,y_train,y_test = train_test_split(
    dataset.data,dataset.target,train_size = 0.8, random_state = 33)

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
# 기존 acc :  0.9385964912280702
# 열 제거 acc :  0.9649122807017544

# Gradientboosting
# 기존 acc :  0.9122807017543859
# 열 제거 acc :  0.9736842105263158

# xgboost
# 기존 acc :  0.9473684210526315


# n_jobs = -1
# 기존 acc :  0.8660078570295027
# 작업 시간 :  0.11466145515441895

# n_jobs = 8
# 기존 acc :  0.8660078570295027
# 작업 시간 :  0.11084461212158203

# n_jobs = 4
# 기존 acc :  0.8660078570295027
# 작업 시간 :  0.11757969856262207

# n_jobs = 1
# 기존 acc :  0.8660078570295027
# 작업 시간 :  0.1495046615600586