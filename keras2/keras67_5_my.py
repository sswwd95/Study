import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import PIL.Image as pilimg
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2
import os

x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_val = np.load('../data/image/gender/npy/keras67_val_x.npy')
y_val = np.load('../data/image/gender/npy/keras67_val_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42, stratify=Y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


model=load_model('../data/h5/fitgen.h5')



loss, acc = model.evaluate(x_val, y_val)
print('loss, acc : ', loss, acc)
# loss, acc :  0.30173632502555847 0.90625

image = pilimg.open('../data/image/me/star3.jpg')
pix = image.resize((64,64))
pix = np.array(pix)
test = pix.reshape(-1,64,64,3)/255.

pred = model.predict(test)
print('pred : ',pred)

for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))

print('여자일 확률은 ', (1-pred)*100, '%')
print('남자일 확률은 ', pred*100, '%')

if pred >0.5:
    print('당신은 남자입니다!')
else:
    print('당신은 여자입니다!')

<<<<<<< Updated upstream:keras2/keras67_5_my.py
# 나
# pred :  [[0.00595943]]
# 여자일 확률은  [[99.40405]] %
# 남자일 확률은  [[0.5959431]] %
# 당신은 여자입니다!

# 영리
# pred :  [[0.00538115]]
# 여자일 확률은  [[99.46188]] %
# 남자일 확률은  [[0.53811514]] %
# 당신은 여자입니다!

# 마동석
# pred :  [[0.99944156]]
# 여자일 확률은  [[0.05584359]] %
# 남자일 확률은  [[99.94415]] %
# 당신은 남자입니다!

=======
def Dataization(img_path):
    image_w = 64
    image_h = 64
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/255)
 
src = []
name = []
test = []
image_dir = "../project/test/"
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))
 
 
test = np.array(test)
model = load_model('Gersang.h5')
predict = model.predict_classes(test)
 
for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))
>>>>>>> Stashed changes:pred.py
