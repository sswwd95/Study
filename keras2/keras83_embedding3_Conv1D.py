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



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()

model.add(Embedding(input_dim=28, output_dim=11, input_length=5)) #(None, 5, 11)  
# model.add(Embedding(28,11))
model.add(Conv1D(32,2))
model.add(Flatten()) # 안해도 먹히긴 하지만 하는게 성능 더 좋다.
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
conv1d (Conv1D)              (None, 4, 32)             736
_________________________________________________________________
dense (Dense)                (None, 4, 1)              33
=================================================================
Total params: 1,077
Trainable params: 1,077
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
# flatten 안 넣을 경우
# 0.8269230723381042

# flatten 넣을 경우
# 1.0