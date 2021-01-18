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

