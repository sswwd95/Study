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

        # model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print (name, '의 정답률 : ',scores)
    except : 
        continue
        print(name,'은 없는 놈!')

'''
AdaBoostClassifier 의 정답률 :  [0.97802198 0.94505495 0.97802198 0.95604396 0.97802198]
BaggingClassifier 의 정답률 :  [0.95604396 0.94505495 0.96703297 0.95604396 0.93406593]
BernoulliNB 의 정답률 :  [0.62637363 0.59340659 0.58241758 0.71428571 0.69230769]
CalibratedClassifierCV 의 정답률 :  [0.86813187 0.94505495 0.95604396 0.93406593 0.92307692]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ComplementNB 의 정답률 :  [0.86813187 0.91208791 0.94505495 0.86813187 0.87912088]
DecisionTreeClassifier 의 정답률 :  [0.94505495 0.91208791 0.92307692 0.93406593 0.91208791]
DummyClassifier 의 정답률 :  [0.59340659 0.45054945 0.46153846 0.6043956  0.50549451]
ExtraTreeClassifier 의 정답률 :  [0.95604396 0.91208791 0.94505495 0.9010989  0.94505495]
ExtraTreesClassifier 의 정답률 :  [0.95604396 0.97802198 0.95604396 0.93406593 0.98901099]
GaussianNB 의 정답률 :  [0.92307692 0.91208791 0.93406593 0.95604396 0.97802198]
GaussianProcessClassifier 의 정답률 :  [0.92307692 0.92307692 0.92307692 0.89010989 0.89010989]
GradientBoostingClassifier 의 정답률 :  [0.97802198 0.93406593 0.94505495 0.92307692 0.97802198]
HistGradientBoostingClassifier 의 정답률 :  [0.96703297 0.96703297 0.96703297 0.96703297 0.97802198]
KNeighborsClassifier 의 정답률 :  [0.91208791 0.93406593 0.93406593 0.83516484 0.93406593]
LabelPropagation 의 정답률 :  [0.32967033 0.43956044 0.40659341 0.28571429 0.41758242]
LabelSpreading 의 정답률 :  [0.32967033 0.37362637 0.41758242 0.40659341 0.35164835]
LinearDiscriminantAnalysis 의 정답률 :  [0.96703297 0.97802198 0.92307692 0.94505495 0.92307692]        
LinearSVC 의 정답률 :  [0.92307692 0.86813187 0.92307692 0.93406593 0.9010989 ]
LogisticRegression 의 정답률 :  [0.92307692 0.92307692 0.97802198 0.96703297 0.94505495]
LogisticRegressionCV 의 정답률 :  [0.94505495 0.94505495 0.96703297 0.96703297 0.95604396]
MLPClassifier 의 정답률 :  [0.89010989 0.96703297 0.91208791 0.94505495 0.9010989 ]
MultinomialNB 의 정답률 :  [0.95604396 0.9010989  0.87912088 0.86813187 0.87912088]
NearestCentroid 의 정답률 :  [0.85714286 0.91208791 0.89010989 0.9010989  0.87912088]
NuSVC 의 정답률 :  [0.89010989 0.9010989  0.81318681 0.85714286 0.9010989 ]
PassiveAggressiveClassifier 의 정답률 :  [0.87912088 0.86813187 0.9010989  0.9010989  0.91208791]       
Perceptron 의 정답률 :  [0.69230769 0.9010989  0.87912088 0.84615385 0.71428571]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.         0.96703297 0.93406593 0.92307692 0.96703297]
RandomForestClassifier 의 정답률 :  [0.94505495 0.94505495 0.97802198 0.98901099 0.92307692]
RidgeClassifier 의 정답률 :  [0.92307692 0.95604396 0.96703297 0.94505495 0.96703297]
RidgeClassifierCV 의 정답률 :  [0.93406593 0.95604396 0.92307692 0.98901099 0.94505495]
SGDClassifier 의 정답률 :  [0.92307692 0.75824176 0.86813187 0.87912088 0.91208791]
SVC 의 정답률 :  [0.92307692 0.89010989 0.89010989 0.95604396 0.89010989]'''

###########################
# Tensorflow
# LSTM 
# acc : 0.9824561476707458
###########################
############################################
# 머신러닝
# RidgeClassifier 의 정답률 :  0.9824561403508771
############################################