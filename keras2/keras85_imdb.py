from tensorflow.keras.datasets import reuters, imdb

(x_train, y_train),(x_test, y_test) = imdb.load_data(
    num_words=5000)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (25000,) (25000,)
# (25000,) (25000,)

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D

model = Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.2))
model.add(Conv1D(64,5, padding='valid', activation='relu',strides=1))
model.add(LSTM(55))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
model.fit(x_train, y_train, batch_size=64, epochs=100,validation_split=0.2)

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss :  1.1877137422561646
# acc :  0.8255199790000916
