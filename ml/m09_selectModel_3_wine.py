from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)

allAlgorithms = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print (name, '의 정답률 : ', accuracy_score(y_test,y_pred))
    except : 
        continue
        print(name,'은 없는 놈!')

'''
AdaBoostClassifier 의 정답률 :  0.8611111111111112
BaggingClassifier 의 정답률 :  0.9444444444444444
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  1.0
CheckingClassifier 의 정답률 :  0.2222222222222222
ComplementNB 의 정답률 :  0.5833333333333334
DecisionTreeClassifier 의 정답률 :  0.9166666666666666
DummyClassifier 의 정답률 :  0.19444444444444445
ExtraTreeClassifier 의 정답률 :  0.9444444444444444
ExtraTreesClassifier 의 정답률 :  0.9722222222222222
GaussianNB 의 정답률 :  0.9722222222222222
GaussianProcessClassifier 의 정답률 :  0.6388888888888888
GradientBoostingClassifier 의 정답률 :  0.9166666666666666
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.6388888888888888
LabelPropagation 의 정답률 :  0.3611111111111111
LabelSpreading 의 정답률 :  0.3611111111111111
LinearDiscriminantAnalysis 의 정답률 :  0.9722222222222222
LinearSVC 의 정답률 :  0.9444444444444444
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultinomialNB 의 정답률 :  0.8888888888888888
NearestCentroid 의 정답률 :  0.7777777777777778
NuSVC 의 정답률 :  0.9166666666666666
PassiveAggressiveClassifier 의 정답률 :  0.6388888888888888
Perceptron 의 정답률 :  0.3888888888888889
QuadraticDiscriminantAnalysis 의 정답률 :  0.9444444444444444
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.5555555555555556
SVC 의 정답률 :  0.6388888888888888
'''       

###########################
# Tensorflow
# acc : 1.0
###########################

############################################
# 머신러닝
# CalibratedClassifierCV 의 정답률 :  1.0
# LinearSVC 의 정답률 :  1.0
# LogisticRegressionCV 의 정답률 :  1.0
# MLPClassifier 의 정답률 :  1.0
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  1.0
# RidgeClassifierCV 의 정답률 :  1.0
############################################