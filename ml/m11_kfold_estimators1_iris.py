from sklearn.model_selection import train_test_split, KFold, cross_val_score
# model_selection에는 모델 전처리하는 기능이 많이 들어감
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 55)

# kfold = KFold(n_splits=5,shuffle=True)
# n_splits는 임의로 정함, shuffle =true는 전체 행을 섞는 것

import sklearn
print(sklearn.__version__) # 0.23.2


allAlgorithms = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        scores = cross_val_score(model, x_train, y_train, cv=5)
        # kfold = KFold(n_splits=4,shuffle=True) => cv=5

        # model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print (name, '의 정답률 : ',scores)
    except : 
        continue
        print(name,'은 없는 놈!')
'''
AdaBoostClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.91666667 1.        ]
BaggingClassifier 의 정답률 :  [0.95833333 1.         0.91666667 0.91666667 1.        ]
BernoulliNB 의 정답률 :  [0.375      0.33333333 0.33333333 0.33333333 0.33333333]
CalibratedClassifierCV 의 정답률 :  [0.91666667 0.95833333 0.875      0.70833333 0.91666667]
CategoricalNB 의 정답률 :  [0.83333333 0.95833333 0.79166667 1.         1.        ]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ComplementNB 의 정답률 :  [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667]
DecisionTreeClassifier 의 정답률 :  [0.95833333 0.91666667 0.91666667 0.91666667 0.95833333]
DummyClassifier 의 정답률 :  [0.25       0.33333333 0.5        0.41666667 0.29166667]
ExtraTreeClassifier 의 정답률 :  [0.91666667 0.95833333 0.875      0.83333333 0.95833333]
ExtraTreesClassifier 의 정답률 :  [0.95833333 1.         0.91666667 0.91666667 0.95833333]
GaussianNB 의 정답률 :  [0.95833333 1.         0.875      0.875      1.        ]
GaussianProcessClassifier 의 정답률 :  [0.95833333 1.         0.91666667 0.95833333 0.95833333]
GradientBoostingClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.91666667 1.        ]
HistGradientBoostingClassifier 의 정답률 :  [0.95833333 0.95833333 0.875      0.95833333 0.95833333]
KNeighborsClassifier 의 정답률 :  [0.95833333 1.         0.91666667 0.95833333 0.95833333]
LabelPropagation 의 정답률 :  [0.95833333 1.         0.95833333 0.91666667 0.95833333]
LabelSpreading 의 정답률 :  [0.95833333 1.         0.95833333 0.91666667 0.95833333]
LinearDiscriminantAnalysis 의 정답률 :  [0.95833333 1.         0.95833333 0.95833333 1.        ]
LinearSVC 의 정답률 :  [0.95833333 0.95833333 0.91666667 0.91666667 1.        ]
LogisticRegression 의 정답률 :  [0.95833333 1.         0.91666667 0.91666667 1.        ]
LogisticRegressionCV 의 정답률 :  [0.95833333 1.         0.91666667 0.95833333 1.        ]
MLPClassifier 의 정답률 :  [0.95833333 1.         1.         1.         0.95833333]
MultinomialNB 의 정답률 :  [0.91666667 0.95833333 1.         0.95833333 0.95833333]
NearestCentroid 의 정답률 :  [0.91666667 1.         0.83333333 0.91666667 0.95833333]
NuSVC 의 정답률 :  [0.95833333 1.         0.875      1.         0.95833333]
PassiveAggressiveClassifier 의 정답률 :  [0.75       1.         1.         0.875      0.70833333]
Perceptron 의 정답률 :  [0.91666667 0.875      0.875      0.58333333 0.91666667]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 1.        ]     
RandomForestClassifier 의 정답률 :  [0.95833333 1.         0.875      0.91666667 0.95833333]
RidgeClassifier 의 정답률 :  [0.79166667 0.95833333 0.83333333 0.70833333 0.875     ]
RidgeClassifierCV 의 정답률 :  [0.79166667 0.95833333 0.83333333 0.70833333 0.875     ]
SGDClassifier 의 정답률 :  [0.66666667 0.83333333 0.875      0.95833333 0.70833333]
SVC 의 정답률 :  [0.95833333 1.         0.875      0.95833333 0.95833333]
'''


###########################
# Tensorflow
# LSTM
# acc : 1.0
###########################

############################################
# 머신러닝
# CategoricalNB 의 정답률 :  1.0
# RadiusNeighborsClassifier 의 정답률 :  1.0
############################################