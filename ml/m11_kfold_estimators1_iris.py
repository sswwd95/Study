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
# n_splits는 임의로 정함, shuffle =true는 전체 행을 섞는 

import sklearn
print(sklearn.__version__) # 0.23.2


allAlgorithms = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        scores = cross_val_score(model, x_train, y_train, cv=5)
        # kfold = KFold(n_splits=4,shuffle=True) => cv=5

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print (name, '의 정답률 : ',scores)
    except : 
        continue
        print(name,'은 없는 놈!')
'''
AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.8333333333333334
CategoricalNB 의 정답률 :  1.0
CheckingClassifier 의 정답률 :  0.3
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9666666666666667
DummyClassifier 의 정답률 :  0.23333333333333334
ExtraTreeClassifier 의 정답률 :  0.9666666666666667
ExtraTreesClassifier 의 정답률 :  0.9666666666666667
GaussianNB 의 정답률 :  0.9666666666666667
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667
LinearSVC 의 정답률 :  0.8666666666666667
LogisticRegression 의 정답률 :  0.9666666666666667
LogisticRegressionCV 의 정답률 :  0.9666666666666667
MLPClassifier 의 정답률 :  0.9333333333333333
MultinomialNB 의 정답률 :  0.9666666666666667
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 :  0.8666666666666667
Perceptron 의 정답률 :  0.6666666666666666
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 :  1.0
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.7333333333333333
RidgeClassifierCV 의 정답률 :  0.7333333333333333
SGDClassifier 의 정답률 :  0.7
SVC 의 정답률 :  0.9666666666666667
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