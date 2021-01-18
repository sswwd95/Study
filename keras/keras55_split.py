'''
#1. split 함수(다:1)
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps) :   
    x, y = list(), list()
    for i in range(len(dataset)) : 
        end_number = i + time_steps
        if end_number > len(dataset) -1 : 
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x , y = split_xy1(dataset,4)
print(x, "\n", y)
'''
'''
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
#  [ 5  6  7  8  9 10]

import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps) :        # 첫 번째 입력값에 데이터셋을 넣어주고, time_steps에 자르고 싶은 컬럼 수를 넣는다. (x,y 나누기)
    x, y = list(), list()                   # 리턴해줄 x와 y를 리스트로 정의
    for i in range(len(dataset)) :          # dataset의 개수만큼 for문을 돌린다. 
        end_number = i + time_steps         # 마지막 숫자가 몇인지를 정의한다. 1회전 할 때는 i는 0이므로 0+4=4, 마지막 숫자는 4가 된다. 
        if end_number > len(dataset) -1 :   # 방금 구한 마지막 숫자가 dataset의 전체 길이에서 1개를 뺀 값보다 크면 for문을 정지한다. end_number가 10이 넘으면 for문이 break 되면서 함수 끝난다.
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]  # y가 0일때 tmp_x는 dataset[0:4]이므로 [1,2,3,4]가 되고, tmp_y는 dataset[4]이므로 다섯번째 숫자인 5가 된다. 
        x.append(tmp_x)                     # 이 값들이 for문을 통해 마지막 숫자(end_number)가 10이 넘지 않을 때까지 반복해서 리스트에 append로 붙게 된다. (tmp는 임시 이름)
        y.append(tmp_y)
    return np.array(x), np.array(y)         # for문이 모두 끝나면 이 함수는 x와 y값을 반환한다. 

x , y = split_xy1(dataset,3)
print(x, "\n", y)

# x데이터를 3개씩 자르면?
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]
#  [7 8 9]] 
#  [ 4  5  6  7  8  9 10]

'''
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
        y_end_number = x_end_number + y_column          # x_end_number가 x의 끝 번호이므로 y_end_number는 y_column의 개수만큼 추가되어 끝자리 나타낸다. 
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
'''
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
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset) : 
            break
        tmp_x = dataset[i:x_end_number, : ]                   #dataset에 리스트로 계속 연결될 때, x는 i가 0일 경우 0부터 4까지의 행, 그 행의 전체 열이 첫 번째 x에 입력되고
        tmp_y = dataset[x_end_number:y_end_number, :]         #이것이 append를 통해서 이어져서 전체 x를 구성한다. 
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(dataset, 3, 1)
print(x,'\n', y)
print(x.shape)
print(y.shape)

# [[[ 1 11 21]
#   [ 2 12 22]
#   [ 3 13 23]]

#  [[ 2 12 22]
#   [ 3 13 23]
#   [ 4 14 24]]

#  [[ 3 13 23]
#   [ 4 14 24]
#   [ 5 15 25]]

#  [[ 4 14 24]
#   [ 5 15 25]
#   [ 6 16 26]]

#  [[ 5 15 25]
#   [ 6 16 26]
#   [ 7 17 27]]

#  [[ 6 16 26]
#   [ 7 17 27]
#   [ 8 18 28]]

#  [[ 7 17 27]
#   [ 8 18 28]
#   [ 9 19 29]]]


#  [[[ 4 14 24]]

#  [[ 5 15 25]]

#  [[ 6 16 26]]

#  [[ 7 17 27]]

#  [[ 8 18 28]]

#  [[ 9 19 29]]

#  [[10 20 30]]]
# (7, 3, 3)
# (7, 1, 3)

# x를 3행 3열로 자르고, y는 3행 2열로 자른다면? x, y = split_xy5(dataset, 3, 2)
