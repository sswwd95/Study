# # 기존 split
# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size +1):
#         subset = seq[i : (i+size)]
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)
'''

############전체 컬럼에서 행을 잘라서 x로 만들고 다음행을 y로 만들고 싶을 때 사용#########
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
        tmp_x = dataset[i:x_end_number, : ]                   
        tmp_y = dataset[x_end_number:y_end_number, :]         
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(dataset, 3, 1)
print(x,'\n', y)
print(x.shape)
print(y.shape)

'''

################# x와 y의 행,열을 다르게 자르고 싶을 때 ####################
def split_xy(dataset,x_row,x_col,y_row,y_col):
    x,y = [],[]
    for i in range(len(dataset)):
        x_end = i + x_row    # x_row = x의 행을 몇개로 자를건지 + i(횟수) => x의 끝번호
        y_end = x_end + y_row  # y의 끝번호는 x의 끝번호에 y 행을 몇개로 자를건지 더한다. 
        if y_end>len(dataset):
            break
        x_tmp = dataset[i:x_end,:x_col]   
        y_tmp = dataset[x_end:y_end,-(y_col):]
        x.append(x_tmp)
        y.append(y_tmp) 
    return np.array(x), np.array(y)
x, y = split_xy(dataset, 3, 2, 1,2) 

print(x, "\n", y)
print("x.shape : ", x.shape) 
print("y.shape : ", y.shape) 

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
################
#  [[[14 24]]

#  [[15 25]]

#  [[16 26]]

#  [[17 27]]

#  [[18 28]]

#  [[19 29]]

#  [[20 30]]]
# x.shape :  (7, 3, 2)
# y.shape :  (7, 1, 2)