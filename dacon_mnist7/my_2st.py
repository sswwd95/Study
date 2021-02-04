import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import cv2 # OpenCV는 단일 이미지나 동영상의 이미지를 원하는 결과를 분석 및 추출하기 위한 API
import gc #Gabage Collection
# 파이썬은 c또는 c++과 같이 프로그래머가 직접 메모리를 관리하지 않고 레퍼런스카운트와 가비지콜렉션에 의해 관리된다
# gc는 메모리의 모든 객체를 추적한다. 새로운 객체는 1세대에서 시작하고 객체가 살아남으면 두번째 세대로 간다. 
# 파이썬의 가비지 수집기는 총 3게대이며, 객체는 현재 세대의 가비지 수집 프로세스에서 살아남을 때마다 이전 세대로 이동한다. 
# 각 세대마다 임계값 개수의 개체가 있는데 객체 수가 해당 임계값을 초과하면 가비지 콜렉션이 콜렉션 프로세스를 추적한다. 
# 임계값 : Threshold. 만약 0~255사이에서 127의 임계값 지정하고 127보다 작으면 모두 0으로, 127보다 크면 모두 255로 값을 급격하게 변화시킨다. 
# 객체 : 어떠한 속성값과 행동을 가지고 있는 데이터, 파이썬의 모든 것들(숫자, 문자, 함수 등)은 여러 속성과 행동을 가지고 있는 데이터다. 
# 왜 Garbage Collection은 성능에 영향을 주나
# 객체가 많을수록 모든 가비지를 수집하는 데 시간이 오래 걸린다는 것도 분명하다.
# 가비지 컬렉션 주기가 짧다면 응용 프로그램이 중지되는 상항이 증가하고 반대로 주기가 길어진다면 메모리 공간에 가비지가 많이 쌓인다.
from keras import backend as bek
train = pd.read_csv('../dacon7/train.csv')

from sklearn.model_selection import train_test_split

x_train = train.drop(['id','digit', 'letter'], axis=1).values # numpy로 바꾸기
x_train = x_train.reshape(-1,28,28,1)

x_train = np.where((x_train<=20)&(x_train!=0),0.,x_train)
# 최소값, 최대값, 혹은 조건에 해당하는 색인(index) 값을 찾기 : np.argmin(), np.argmax(), np.where()
# 20이하 값 & 0이 아닌 값 => 0으로 바꾸고 아닌것은 그대로 두라는 조건문

# 흑백 이미지 : 대부분 8bit(화소 하나의 색 표현에 8bit 사용), 각 화소의 화소값은 2**8=256개의 값들 중 하나의 값.
# 즉, 0과 255사이의 값들 중 하나의 값이 된다.(0 = 검정색, 255 흰색)
# RGB : 2**8*2**8*2**8 = 2**24 = 16,777,216가지
x_train = x_train/255.
# == x_train = x_train.astype('float32') 실수형으로 바꾸기

y = train['digit']
y_train = np.zeros((len(y),len(y.unique()))) # 총 행의수 , 10(0~9)
# np.zeros : 0으로 초기화된 shape 차원의 ndarray 배열 객체를 반환 

for i, digit in enumerate(y):
    y_train[i,digit]=1
# 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 때 사용
# 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
# i, digit로 쓰면 인덱스와 원소를 각각 다른 변수에 할당(인자 풀기)

# 300x300의 grayscale 이미지로 리사이즈
train_224 = np.zeros([2048,300,300,3],dtype=np.float32)

for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    # converted =  변환 , gray색으로 변환
    resized = cv2.resize(converted,(300,300),interpolation=cv2.INTER_CUBIC)
    # 원본이미지, 결과 이미지 크기, 보간법(cv2.INTER_CUBIC, cv2.INTER_LINEAR 이미지 확대할 때 사용/cv2.INTER_AREA는 사이즈 줄일 때 사용)
    # 보간법(interpolation)이란 통계적 혹은 실험적으로 구해진 데이터들(xi)로부터, 
    # 주어진 데이터를 만족하는 근사 함수(f(x))를 구하고,  이 식을 이용하여 주어진 변수에 대한 함수 값을 구하는 일련의 과정을 의미