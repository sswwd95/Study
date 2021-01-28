from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)
kfold = KFold(n_splits=5,shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print (name, '의 정답률 : ',scores)
    except : 
        continue
        print(name,'은 없는 놈!')

'''
AdaBoostClassifier 의 정답률 :  0.956140350877193
BaggingClassifier 의 정답률 :  0.9385964912280702
BernoulliNB 의 정답률 :  0.5701754385964912
CalibratedClassifierCV 의 정답률 :  0.9385964912280702
CheckingClassifier 의 정답률 :  0.4298245614035088
ComplementNB 의 정답률 :  0.8947368421052632
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.5263157894736842
ExtraTreeClassifier 의 정답률 :  0.9122807017543859
ExtraTreesClassifier 의 정답률 :  0.956140350877193
GaussianNB 의 정답률 :  0.9385964912280702
GaussianProcessClassifier 의 정답률 :  0.9122807017543859
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.9649122807017544
KNeighborsClassifier 의 정답률 :  0.9385964912280702
LabelPropagation 의 정답률 :  0.4649122807017544
LabelSpreading 의 정답률 :  0.4649122807017544
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.9385964912280702
LogisticRegression 의 정답률 :  0.9210526315789473
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9473684210526315
MultinomialNB 의 정답률 :  0.9035087719298246
NearestCentroid 의 정답률 :  0.9035087719298246
NuSVC 의 정답률 :  0.8859649122807017
PassiveAggressiveClassifier 의 정답률 :  0.9122807017543859
Perceptron 의 정답률 :  0.9122807017543859
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RandomForestClassifier 의 정답률 :  0.956140350877193
RidgeClassifier 의 정답률 :  0.9824561403508771
RidgeClassifierCV 의 정답률 :  0.9649122807017544
SGDClassifier 의 정답률 :  0.9122807017543859
SVC 의 정답률 :  0.9210526315789473
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