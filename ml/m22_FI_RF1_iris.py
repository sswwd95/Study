# RandomForest 모델 사용해서 중요도 낮은 피처 제외하고 기존 자료와 acc 비교
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model = RandomForestClassifier(max_depth = 4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

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
print(cut_columns(model.feature_importances_,dataset.feature_names,1))
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

dataset = load_iris()
x = dataset.data
y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names) 
df1 = df.iloc[:,[0,2,3]].values
names = dataset.feature_names[0],dataset.feature_names[2],dataset.feature_names[3]
print(names)

# 1. 데이터
# dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    df1, dataset.target, train_size=0.8, random_state=44
)

# 2. 모델
model=RandomForestClassifier(max_depth = 4) # 깊이 4

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가,예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

print(dataset.feature_names)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_df1(model) : 
    n_features = df1.data.shape[1] #datset shape (150,4)
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features))
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_df1(model)
plt.show()

# 기존 acc :  0.8666666666666667
# acc :  0.9666666666666667