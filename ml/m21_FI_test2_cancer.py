
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

dataset = load_breast_cancer()
x = dataset['data'] # => x = dataset.data
y = dataset['target'] # => y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names) # df = pd.DataFrame(x, columns=dataset.feature_names)
df1 = df.iloc[:,[22,23,24,28]].values
names = dataset.feature_names[[22,23,24,28]]
print(df1.shape)

# 1. 데이터
# dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    df1, dataset.target, train_size=0.8, random_state=44
)

print(dataset.data.shape) #(569, 30)

# 2. 모델
model=DecisionTreeClassifier(max_depth=4) # 깊이 4

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
    plt.yticks(np.arange(n_features), names)
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_df1(model)
plt.show()
'''
[0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.00677572 0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.05612587 0.78678449 0.01008994
 0.02293065 0.         0.         0.11729332 0.         0.        ]
acc :  0.9385964912280702
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
'''

'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model = DecisionTreeClassifier(max_depth = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ', acc)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

####### 여기까지가 기존!!

####### 30개 동안 0이 아닌칼럼만 더해줌!!
####### 피쳐네임은 피쳐라는 리스트에 더해줌!!!
fi = model.feature_importances_
new_data = []
feature = []
for i in range(len(fi)):
    if fi[i] != 0:
        new_data.append(dataset.data[:,i])
        feature.append(dataset.feature_names[i])

new_data = np.array(new_data)
new_data = np.transpose(new_data)
x_train,x_test,y_train,y_test = train_test_split(new_data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model = DecisionTreeClassifier(max_depth = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ', acc)

####### dataset >> new_data 로 바꾸고 featurename 부분을 feature 리스트로 바꿔줌!!!
def plot_feature_importances_dataset(model):
    n_features = new_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), feature)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
'''