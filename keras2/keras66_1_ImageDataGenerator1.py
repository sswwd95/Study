import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# 이미지를 데이터로 전처리. 증폭이 가능하다.
# 다양한 방법 중 하나
# 파라미터도 자동으로 빼준다. 랜덤서치보다는 속도가 빠름. 
# 오토케라스에서는 레이어도 자동. 
# https://keras.io/ko/preprocessing/image/

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 전처리개념
    horizontal_flip=True,  # True하면 가로로 반전
    vertical_flip=True, # True하면 세로로 반전
    width_shift_range=0.1, # 수평이동
    height_shift_range=0.1, # 수직이동
    rotation_range=5, # 회전
    zoom_range=1.2, # 확대, 축소
    shear_range=0.7, # 크게 하면 더 비스듬하게 찌그러진 이미지가 된다. 
    fill_mode='nearest' # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식.  
                        # { "constant", "nearest", "reflect"또는 "wrap"} 중 하나
                        # 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                        # 'nearest': aaaaaaaa|abcd|dddddddd
                        # 'reflect': abcddcba|abcd|dcbaabcd
                        # 'wrap': abcdabcd|abcd|abcdabcd
                        # 0으로하면 빈자리를 0으로 채워준다(padding과 같은 개념)? -> 넣어보고 체크하기
)


test_datagen=ImageDataGenerator(rescale=1./255)
# test는 왜 rescale만 할까? 테스트에선 이미지 증폭을 할 필요가 없다. 조작개념과는 다르다. 
# 훈련할 때는 많은 데이터가 있어야 하기때문에 
# rescale=1./255 -> 이미지는 0~255개의 값을 가진다. 3개 있으면 컬러. 1개면 흑백.

# ----------------------------------이미지를 이렇게 만들겠다고 이름만 선언---------------------------------------------------------

# flow 또는 flow_from_directory 에서 데이터로 변환시켜준다. 
# 
# ad폴더(치매에 걸린 뇌 데이터)는 0으로 하고, normal(정상뇌)은 1로 설정.

#train_generater
#flow_from_directory : 폴더안의 경로 잡아주고 설정. directory(폴더)에 있는 파일을 그대로 가져다 flow시키겠다.
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),  # 가로 150,세로150의 이미지 크기로 만들겠다는 것. 임의로 설정
    batch_size=5,
    class_mode='binary'
)
# Found 160 images belonging to 2 classes.(80장씩 2개의 폴더에 있는 것-ad, normal)

# x의 shape은 (80,150,150,1) -> x값은 0   (80장, 150*150크기, 흑백(1))
# y의 shape은 (80,) -> y값은 1
# 원래 x는 0~255사이인데 rescale하면서 0~1사이로 되는 것.
# 따로 설정을 안해도 clsass mode = binary ->  앞의 값이 0이면 나머지는 1이된다. 


# flow는 수치화되어있는 데이터면 사용.

#test_generater
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150), 
    batch_size=5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

# flow_from_directory에서 나오면 자동으로 x, y train 생성됨.
print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002144003FC70>
# 사이킷런도 x와 y를 같이 싸고 있다. 
print(xy_train[0])
# 0번째를 출력하면 x와 y가 같이 들어가있다. y가 5개. 왜? batch_size를 5로 했기 때문
'''
(array([[[[0.0627451 , 0.0627451 , 0.0627451 ],
         [0.0627451 , 0.0627451 , 0.0627451 ],
         [0.0617597 , 0.0617597 , 0.0617597 ],
         ...,
         [0.06677356, 0.06677356, 0.06677356],
         [0.07282551, 0.07282551, 0.07282551],
         [0.07887746, 0.07887746, 0.07887746]],
         ....
        [[0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.        ],
         ...,
         [0.04177788, 0.04177788, 0.04177788],
         [0.0471611 , 0.0471611 , 0.0471611 ],
         [0.04825744, 0.04825744, 0.04825744]]]], dtype=float32), 
         array([1., 1., 0., 0., 1.], dtype=float32)) # array의 [1]이 y값
'''

print(xy_train[0][0]) # x값만 나온다.첫 번째의 array라서 [0]
print(xy_train[0][0].shape)
#(5, 150, 150, 3)
print(xy_train[0][1]) # y값만 나온다.두 번째의 array[1]이 y값
# [0. 1. 0. 1. 0.] -> 5개
print(xy_train[0][1].shape)
'''
# batch size=10
# (10, 150, 150, 3)
# [0. 1. 0. 1. 1. 0. 1. 0. 0. 0.]
# (10,)
print(xy_train[15][1])
#[0. 1. 0. 1. 0. 0. 1. 1. 0. 1.]
print(xy_train[16][1])
# Asked to retrieve element 16, but the Sequence has length 16
# 160을 10으로 나눈거라서 16번째는 없다. 
'''

# 전체 쉐입 보고싶다면 train의 batch_size=160하기
# x, y 추출가능
# 배치사이즈 모르겠다면 큰 값 주고 찾아보기. 총 데이터보다 큰 값을 줘도 데이터만큼만 나온다. 

'''
만약 batch size를 159개로 한다면? 나머지 1개가 잘리는데 어떻게 연산??
print(xy_train[1][1]) -> [0.]
print(xy_train[1][1].shape) -> (1,)
* fit 으로 할 때는 numpy로 통째로 뽑아서 하는거라 잘릴일이 없다.
* fit generate하면 상관없다. 잘려도 알아서 연산해줌.
'''