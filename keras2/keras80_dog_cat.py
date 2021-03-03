# 이미지는 data/image/vgg/에 넣기(고양이, 강아지, 라이언, 슈트)
# 파일명 : dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jpg',target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg',target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.jpg',target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg',target_size=(224,224))

# print(img_cat) #<PIL.Image.Image image mode=RGB size=224x224 at 0x18D0CD90B80>
# plt.imshow(img_cat)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

print(arr_dog) # 수치화
#  [[ 92.  96.  97.]
#   [ 91.  95.  96.]
#   [ 96. 100. 101.]
#   ...
#   [242. 231. 229.]
#   [241. 230. 228.]
#   [242. 231. 229.]]]
print(type(arr_cat)) #<class 'numpy.ndarray'>
print(arr_dog.shape) #(224, 224, 3)

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
# img_arr한거를 알아서 모양 맞춰준다.
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog)
#  [[128.061 117.221 111.32 ]
#   [127.061 116.221 110.32 ]
#   [127.061 116.221 110.32 ]
#   ...
#   [131.061 120.221 114.32 ]
#   [130.061 119.221 113.32 ]
#   [130.061 119.221 113.32 ]]]
print(arr_dog.shape) #(224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
# np.stack => 순서대로 연결
print(arr_input.shape) #(4, 224, 224, 3)

# 2. 모델구성
model = VGG16()
results = model.predict(arr_input)

print(results)
#  [1.32918819e-06 2.08956044e-05 1.69381235e-06 ... 1.71409226e-06
#   1.56086790e-05 1.14980270e-03]
#  [1.03225811e-05 1.14807517e-06 1.00580405e-06 ... 2.58051728e-06
#   1.78455866e-05 7.71751365e-05]]
print('results.shape : ', results.shape)
# results.shape :  (4, 1000) -> 1000? 이미지넷에서 분류할 수 있는 카테고리의 수

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions #예측한 것을 해석하겠다

decode_results = decode_predictions(results)
print('++++++++++++++++++++++++++++++++++++')
print('results[0] : ', decode_results[0])
print('++++++++++++++++++++++++++++++++++++')
print('results[1] : ', decode_results[1])
print('++++++++++++++++++++++++++++++++++++')
print('results[2] : ', decode_results[2])
print('++++++++++++++++++++++++++++++++++++')
print('results[3] : ', decode_results[3])
'''
++++++++++++++++++++++++++++++++++++
results[0] :  [('n02113624', 'toy_poodle', 0.6611787), ('n02113712', 'miniature_poodle', 0.33761844), ('n02113799', 'standard_poodle', 0.001004679), ('n02102480', 'Sussex_spaniel', 5.4335746e-05), ('n02102318', 'cocker_spaniel', 4.0798994e-05)]
++++++++++++++++++++++++++++++++++++
results[1] :  [('n02123045', 'tabby', 0.6405649), ('n02123159', 'tiger_cat', 0.2624627), ('n02124075', 'Egyptian_cat', 0.015971713), ('n04265275', 'space_heater', 0.010649668), ('n03887697', 'paper_towel', 0.007252967)]
++++++++++++++++++++++++++++++++++++
results[2] :  [('n03291819', 'envelope', 0.21738482), ('n02786058', 'Band_Aid', 0.091208614), ('n03598930', 'jigsaw_puzzle', 0.06482566), ('n03908618', 'pencil_box', 0.058071256), ('n06359193', 'web_site', 0.05694257)]
++++++++++++++++++++++++++++++++++++
results[3] :  [('n02883205', 'bow_tie', 0.19497494), ('n04350905', 'suit', 0.19430502), ('n03763968', 'military_uniform', 0.10645677), ('n04479046', 'trench_coat', 0.06444714), ('n03124170', 'cowboy_hat', 0.041933198)]
'''