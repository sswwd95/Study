import numpy as np

a = np.array(range(1,11))
size = 5

print(a)

# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size +1):
#         subset = seq[i : (i+size)]
#         aaa.append([item for item in subset])
#     print(type(aaa))
#     return np.array(n)

# dataset = split_x(a, size)
# print("=======================")
# print(dataset)

# append 함수는 선택한 요소의 내용의 끝에 새로운 콘텐츠를 추가할 수 있다
# for i in -> range 안에 i가 있으면 계속 반복
# aaa는 임의지정

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=======================")
print(dataset)

# [ 1  2  3  4  5  6  7  8  9 10]
# <class 'list'>
# =======================
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
