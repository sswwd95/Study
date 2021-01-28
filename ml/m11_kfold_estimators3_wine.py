from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)
kfold = KFold(n_splits=4,shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        # model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print (name, '의 정답률 : ',scores)
    except : 
        continue
        print(name,'은 없는 놈!')

'''
AdaBoostClassifier 의 정답률 :  [1.         1.         0.85714286 1.         0.85714286]
BaggingClassifier 의 정답률 :  [1.         0.96551724 0.96428571 0.92857143 0.85714286]
BernoulliNB 의 정답률 :  [0.31034483 0.44827586 0.35714286 0.39285714 0.28571429]
CalibratedClassifierCV 의 정답률 :  [0.89655172 0.89655172 0.92857143 0.89285714 0.85714286]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ComplementNB 의 정답률 :  [0.5862069  0.72413793 0.71428571 0.67857143 0.64285714]
DecisionTreeClassifier 의 정답률 :  [0.89655172 0.96551724 0.92857143 1.         0.85714286]
DummyClassifier 의 정답률 :  [0.24137931 0.24137931 0.25       0.35714286 0.28571429]
ExtraTreeClassifier 의 정답률 :  [0.75862069 0.82758621 0.85714286 0.85714286 0.82142857]
ExtraTreesClassifier 의 정답률 :  [0.96551724 1.         1.         1.         0.96428571]
GaussianNB 의 정답률 :  [0.96551724 1.         0.89285714 1.         0.92857143]
GaussianProcessClassifier 의 정답률 :  [0.55172414 0.4137931  0.39285714 0.35714286 0.39285714]
GradientBoostingClassifier 의 정답률 :  [0.93103448 0.93103448 1.         1.         0.96428571]
HistGradientBoostingClassifier 의 정답률 :  [0.96551724 0.96551724 1.         0.92857143 0.92857143]
KNeighborsClassifier 의 정답률 :  [0.68965517 0.68965517 0.75       0.64285714 0.71428571]
LabelPropagation 의 정답률 :  [0.44827586 0.51724138 0.39285714 0.53571429 0.53571429]
LabelSpreading 의 정답률 :  [0.44827586 0.4137931  0.57142857 0.42857143 0.5       ]
LinearDiscriminantAnalysis 의 정답률 :  [1.         0.96551724 0.96428571 1.         1.        ]
LinearSVC 의 정답률 :  [0.48275862 0.44827586 0.92857143 0.78571429 0.78571429]
LogisticRegression 의 정답률 :  [0.89655172 0.93103448 0.82142857 0.96428571 0.96428571]
LogisticRegressionCV 의 정답률 :  [1.         0.93103448 1.         0.89285714 0.89285714]
MLPClassifier 의 정답률 :  [0.48275862 0.37931034 0.57142857 0.35714286 0.39285714]
MultinomialNB 의 정답률 :  [0.93103448 0.89655172 0.82142857 0.89285714 0.82142857]
NearestCentroid 의 정답률 :  [0.72413793 0.75862069 0.64285714 0.60714286 0.78571429]
NuSVC 의 정답률 :  [0.96551724 0.89655172 0.75       0.89285714 0.82142857]
PassiveAggressiveClassifier 의 정답률 :  [0.65517241 0.37931034 0.32142857 0.5        0.21428571]
Perceptron 의 정답률 :  [0.62068966 0.5862069  0.75       0.39285714 0.64285714]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.96551724 1.         1.         1.         1.        ]
RandomForestClassifier 의 정답률 :  [0.96551724 1.         1.         0.92857143 1.        ]
RidgeClassifier 의 정답률 :  [0.96551724 1.         1.         1.         1.        ]
RidgeClassifierCV 의 정답률 :  [0.96551724 0.96551724 0.96428571 0.96428571 1.        ]
SGDClassifier 의 정답률 :  [0.68965517 0.65517241 0.60714286 0.89285714 0.71428571]
SVC 의 정답률 :  [0.68965517 0.65517241 0.57142857 0.67857143 0.75      ]
'''

###########################
# Tensorflow
# LSTM 
# acc : 0.9824561476707458
###########################
############################################
# 머신러닝
# RidgeClassifier 의 정답률 :  0.9824561403508771
############################################