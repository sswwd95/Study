'''
# 2. split함수 (다:다) 
# 내일과 모레의 주가를 예측하는 모델 만들기 가능 (주식, 환율, 유가, 선물, 파생상품 등)
# 3개의 time_steps로 x를 만들고 2개의 y컬럼을 만들기

import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy2(dataset, time_steps, y_column) :          # y값의 원하는 열의 개수 지정
    x, y =list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps        
        y_end_number = x_end_number + y_column          # x_end_number가 x의 끝이므로 y_end_number는 y_column의 개수만큼 추가되어 끝자리 나타낸다. 
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 4
y_column = 2
x, y = split_xy2(dataset, time_steps, y_column)
print(x, "\n", y)
print('x.shape : ', x.shape)
print('y.shape : ', y.shape)

# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]] 
#  [[ 5  6]
#  [ 6  7]
#  [ 7  8]
#  [ 8  9]
#  [ 9 10]]
# x.shape :  (5, 4)
# y.shape :  (5, 2)

'''
'''
# split 다입력(다:1) -> 다음날 예측
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print('dataset.shape : ' , dataset.shape)
# dataset.shape :  (3, 10)

dataset = np.transpose(dataset)      # 컬럼이 여러 개인 데이터를 시계열에 쓰기 좋도록 자르는 작업
print(dataset)
print('dataset.shape : ', dataset.shape)
# [[ 1 11 21]
#  [ 2 12 22]
#  [ 3 13 23]
#  [ 4 14 24]
#  [ 5 15 25]
#  [ 6 16 26]
#  [ 7 17 27]
#  [ 8 18 28]
#  [ 9 19 29]
#  [10 20 30]]
# dataset.shape :  (10, 3)

def split_xy3(dataset,time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]    # :-1은 y열을 제외하고 앞까지
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x,y = split_xy3(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)
# [[[ 1 11]
#   [ 2 12]
#   [ 3 13]]

#  [[ 2 12]
#   [ 3 13]
#   [ 4 14]]

#  [[ 3 13]
#   [ 4 14]
#   [ 5 15]]

#  [[ 4 14]
#   [ 5 15]
#   [ 6 16]]

#  [[ 5 15]
#   [ 6 16]
#   [ 7 17]]

#  [[ 6 16]
#   [ 7 17]
#   [ 8 18]]

#  [[ 7 17]
#   [ 8 18]
#   [ 9 19]]

#  [[ 8 18]
#   [ 9 19]
#   [10 20]]]
#  [[23]
#  [24]
#  [25]
#  [26]
#  [27]
#  [28]
#  [29]
#  [30]]
# (8, 3, 2)
# (8, 1) 

y = y.reshape(y.shape[0]) # 마지막 아웃풋 shape을 맞춰주려면 벡터를 스칼라형태로 맞춰주기 위해 reshape해준다. 
print(y.shape) #(8,)
'''

# split 다입력(다:다) -> 향후 2일 뒤 예측

import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print('dataset.shape : ' , dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print('dataset.shape : ', dataset.shape)

def split_xy3(dataset,time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]    # :-1은 y열을 제외하고 앞까지
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x,y = split_xy3(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)
print(y.shape)

# [[[ 1 11]
#   [ 2 12]
#   [ 3 13]]

#  [[ 2 12]
#   [ 3 13]
#   [ 4 14]]

#  [[ 3 13]
#   [ 4 14]
#   [ 5 15]]

#  [[ 4 14]
#   [ 5 15]
#   [ 6 16]]

#  [[ 5 15]
#   [ 6 16]
#   [ 7 17]]

#  [[ 6 16]
#   [ 7 17]
#   [ 8 18]]

#  [[ 7 17]
#   [ 8 18]
#   [ 9 19]]]
#  [[23 24]
#  [24 25]
#  [25 26]
#  [26 27]
#  [27 28]
#  [28 29]
#  [29 30]]
# (7, 3, 2)
# (7, 2)

# 5.split 다입력(다:다)
# 전체 컬럼에서 행으로 잘라서 x를 만들고, 그 다음 행을 y로 만드는 데이터 형식의 함수. 

import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])
print('dataset.shape : ' , dataset.shape)

dataset = np.transpose(dataset)
print(dataset)
print('dataset.shape : ', dataset.shape)

def split_xy5(dataset, time_steps, y_column) : 
    x, y = list(), list()
    for i in range(len(dataset)) : 
        x_end_number = i + time_steps
        y_end_number = time_stepas + y_column

        if y_end_number > len(dataset) : 
            break
        tmp_x = dataset[i:x_end_number, : ]
        tmp_y = dataset