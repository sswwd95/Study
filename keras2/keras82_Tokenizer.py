from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # 순서대로(빈도수 높은 단어가 앞으로)
# {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

# 텍스트를 토큰화
x = token.texts_to_sequences([text])
print(x)
# [[3, 1, 1, 4, 5, 1, 2, 2, 6]]

from tensorflow.keras.utils import to_categorical 
word_size = len(token.word_index) # 1부터 시작
print(word_size) #6

x = to_categorical(x) # 0부터 시작

print(x)
# [[[0. 0. 0. 1. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1.]]]
print(x.shape) #(1, 9, 7)

print(token.document_count) #1
# 총 몇 개의 문장이 들어있는지 셀 수 있다