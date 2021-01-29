
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
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

def plot_feature_importances_dataset(model) : 
    n_features = dataset.data.shape[1] #datset shape (150,4)
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
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