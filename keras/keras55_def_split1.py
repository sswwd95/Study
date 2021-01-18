import numpy as np

# # 기존 split

# a = np.array(range(1,11))
# size = 5

# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size +1):
#         subset = seq[i : (i+size)]
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)
# dataset = split_x(a, size)
# print("=======================")
# print(dataset)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# (dataset,x_len,y_len)
# 전체 데이터에서 4개의 time_steps로 x만들고 2개의 y컬럼 맞춰준다. 
import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy2(dataset, time_steps, y_column) :          
    x, y =list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps        
        y_end_number = x_end_number + y_column         
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