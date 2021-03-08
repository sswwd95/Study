import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255

# 노이즈있는 데이터를 임의로 생성

x_train_noised = x_train + np.random.normal(0,0.1, size=x_train.shape)
# x_train shape에 맞춰서 0부터 0.1사이를 랜덤값으로 .(노이즈)을 찍어준다.
# x_train의 범위는 하얀 숫자부분이 255 숫자 아닌 검은색 부분이 0. 하얀 숫자 부분을 255로 나눠서 1이됨.
# 0~0.1까지이기 때문에 스케일링한 값을 더하면 0~1.1이 된다. => 수치가 0~0.1사이로 더해진 것이기 때문에 전체적으로 밝아진다. (숫자가 커질 수록 밝아진다.)
x_test_noised = x_test + np.random.normal(0,0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) 
# np.clip은 고정시키는 것. 0~1.1을 다시 0~1로 맞춰준다. 1을 넘어가면 1로 맞춰준다.
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoer(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units =784, activation='sigmoid'))
    return model

model = autoencoer(hidden_layer_size=154)

####################### 왜 154? ###################################
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(x_train)  # PCA 계산 후 투영
print('선택한 차원(픽셀) 수 :', pca.n_components_) 
#선택한 차원(픽셀) 수 : 154
####################################################################

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc') # metrics 안해도 된다.

model.fit(x_train_noised, x_train, epochs=10) # 노이즈 있는 것과 없는 것을 비교

output = model.predict(x_test_noised) # x_test에 대한 노이즈가 제거되었는지 확인

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7, ax8, ax9, ax10),  (ax11,ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5, figsize=(20,7))

# 이미지 5개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('input', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('noise', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('output', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
