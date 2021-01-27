from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)

allAlgorithms = all_estimators(type_filter='classifier')
########################################################################################################
'''
# estimator = 추정량
# type_filter='classifier' 분류형 모델 전체를 넣은 것
for (name, algorithm) in allAlgorithms : # all_estimatoers 에서 인자가 2개 나간다. name = 모델의 이름
    model = algorithm() # 전체 모델 다 순차적으로 돌아감

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print (name, '의 정답률 : ', accuracy_score(y_test,y_pred))


AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.8333333333333334
CategoricalNB 의 정답률 :  1.0
CheckingClassifier 의 정답률 :  0.3
Traceback (most recent call last):

TypeError: __init__() missing 1 required positional argument: 'base_estimator'

# import sklearn
# print(sklearn.__version__) # 0.23.2
# -> 버전때문에 base estimator모델에서 막힘
'''
##########################################################################################################
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print (name, '의 정답률 : ', accuracy_score(y_test,y_pred))
    except : 
        # continue
        print(name,'은 없는 놈!')

# 버전 때문에 안돌아가는 모델은 except로 제외시키고 다시 for문으로 들어가서 실행시킴.
'''
AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.8333333333333334
CategoricalNB 의 정답률 :  1.0
CheckingClassifier 의 정답률 :  0.3
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9666666666666667
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9
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
MLPClassifier 의 정답률 :  0.9
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  0.9666666666666667
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
Perceptron 의 정답률 :  0.6666666666666666
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 :  1.0
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.7333333333333333
RidgeClassifierCV 의 정답률 :  0.7333333333333333
SGDClassifier 의 정답률 :  0.8333333333333334
SVC 의 정답률 :  0.9666666666666667
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
#####################################################################################
# continue하면 제외시킨 모델은 빼고 출력
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