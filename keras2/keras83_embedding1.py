from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', ' 한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '별이는 완전 귀여워요']

# 긍정1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
'''
{'참': 1, '너무': 2, '재밌어요': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, 
'싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, ' 
별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미 
없다': 23, '재밌네요': 24, '별이는': 25, '완전': 26, '귀여워요': 27}
'''

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27]]# 크기가 일정하지 않아 모델링 할 수 없다. 길이를 일정하게 맞춰주려면? 긴 글자에 맞게 하고 빈자리는 0으로 채운다.

from tensorflow.keras.preprocessing.sequence import pad_sequences # 2차, 3차 가능
pad_x = pad_sequences(x, padding='pre') # 앞쪽을 0
# pad_x = pad_sequences(x, padding='post') # 뒤쪽을 0

print(pad_x) 
# pre
'''
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  1  5  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]
 [ 0  0 25 26 27]]
'''
# post 
'''
[[ 0  0  2  3]
 [ 0  0  1  4]
 [ 1  5  6  7]
 [ 0  8  9 10]
 [12 13 14 15]
 [ 0  0  0 16]
 [ 0  0  0 17]
 [ 0  0 18 19]
 [ 0  0 20 21]
 [ 0  0  0 22]
 [ 0  0  2 23]
 [ 0  0  1 24]
 [ 0 25 26 27]]
 '''
print(pad_x.shape) #(13, 5)

pad_x = pad_sequences(x,maxlen=5, truncating='pre',padding='pre') 
# maxlen => 내가 원하는 길이로 자른다
# truncating='pre'  : 앞쪽을 자른다
# truncating='post' : 뒷쪽을 자른다

print(pad_x)

print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27]
print(len(np.unique(pad_x))) #28 
# 0부터 27까지인데 11이 maxlen=4로 인해 잘렸다.


'''
원핫인코딩을 그대로 사용하면 벡터의 길이가 너무 길어진다.
만약 만 개의 단어 토큰으로 이루어진 말뭉치를 다룬다고 할 때, 
벡터화 하면 9,999개의 0과 하나의 1로 이루어진 단어 벡트를 1만개를 만들어야 한다.
이러한 공간적 낭비를 해결하기 위해 등장한 것이 단어 임베딩
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()

# model.add(Embedding(input_dim=28, output_dim=11, input_length=5)) 
model.add(Embedding(28,11))
# output_dim=11 : 다음 레이어로 넘겨주는 노드의 갯수. 임의로 지정 가능. 실제단어의 갯수
# input_length=5 : pad_x.shape의 뒤에 있는 값
# embedding함수가 np.unique(pad_x) 계산하기 때문에 27로 넣으면 오류. 실제단어의 갯수보다 같거나 커야한다.
# Embedding은 3차로 받아서 3차로 나간다. 
model.add(LSTM(32))
# LSTM은 3차로 받아서 2차로 나간다.
model.add(Dense(1, activation='sigmoid'))
# model.add(Flatten())
# model.add(Dense(1))


model.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
flatten (Flatten)            (None, 55)                0
_________________________________________________________________
dense (Dense)                (None, 1)                 56
=================================================================
Total params: 364
Trainable params: 364
Non-trainable params: 0
_________________________________________________________________
'''

# model.add(Embedding(28,11))
'''
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 11)          308
=================================================================
Total params: 308
Trainable params: 308
Non-trainable params: 0
_________________________________________________________________
'''
# param 왜 308? input_dim*output_dim (28*11 = 308)
# input_length의 영향이 없다.

# model.add(LSTM(32))
'''
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,973
Trainable params: 5,973
Non-trainable params: 0
_________________________________________________________________

'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
# 1.0