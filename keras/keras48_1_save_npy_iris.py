# 딕셔너리는 키(key)와 값(value)이 쌍으로 이루어진 자료 구조. 
# 리스트나 튜플처럼 순차적으로(Sequential) 해당 요소값을 구하지 않는 것이 특징

from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)
'''
##################### 중요 #######################
{'data': array([[5.1, 3.5, 1.4, 0.2],... -> float형
'target': array([0, 0, 0, 0, 0,          -> int형
'frame': None, 
'target_names': array(['setosa', 'versicolor', 'virginica'], -> y값에 대한 분류
'filename': 'C:\\Users\\bit\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv'}
* csv 파일  : 몇 가지 필드를 쉼표(,)로 구분한 텍스트 데이터 및 텍스트 파일
###################################################
'''
print(dataset.frame)
print(dataset.target)
'''
기존 데이터 불러오는 법
x = dataset.data
y = dataset.target
아래와 같이 써도 된다. 
x = dataset['data'] -> [data]는 안된다. 스트링형태와 같이 들어가야함.
y = dataset['target']
'''
x_data = dataset.data
y_data = dataset.target

print(type(x_data),type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('../data/npy/iris_x.npy',arr=x_data)
np.save('../data/npy/iris_y.npy',arr=y_data)
