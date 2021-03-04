# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.

# mae 0.12이하
import csv
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer): # shuffle_buffer : random state같은 느낌
    series = tf.expand_dims(series, axis=-1) # 열 기준으로 차원 늘려준다. ex(2000,) ->(2000,1)
    ds = tf.data.Dataset.from_tensor_slices(series) # 위의 series를 데이터셋 형식으로 만들어준다.
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) 
    # window_size +1 : 데이터를 몇개씩 자를것인지(+1은 y), shift=1 처음 시작점을 어느정도 띄워서 할 것인가, drop_remainder = True : 데이터 길이가 맞는 것만 사용
    ds = ds.flat_map(lambda w: w.batch(window_size + 1)) # 31개짜리의 데이터 묶음으로 만들어준다.
    ds = ds.shuffle(shuffle_buffer) # 순서를 shuffle
    ds = ds.map(lambda w: (w[:-1], w[1:])) # x = (30,1), y =(30,1)로 잘라준다. 

    # for x in ds :
    #     print(x)
    
    return ds.batch(batch_size).prefetch(1) # batch size로 묶어준다.


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, './/tf_certificate//Category5/sunspots.csv')

    time_step = []
    sunspots = []

    with open('.//tf_certificate//Category5/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2])) 
        time_step.append(int(row[0])) 

    series = np.array(sunspots)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000 # train의 양


    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000


    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(60,5,padding='causal', activation='relu', input_shape=[None,1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
        tf.keras.layers.Dense(1)
    ])
   
    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #


    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    model.compile(loss='mae', optimizer='adam', metrics='mae')
    model.fit(train_set, batch_size=32, epochs=100)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save(".//tf_certificate//Category5/mymodel.h5")



# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
# def model_forecast(model, series, window_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(32).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast


# window_size = 30
# rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

# result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

# # To get the maximum score, your model must have an MAE OF .12 or less.
# # When you Submit and Test your model, the grading infrastructure
# # converts the MAE of your model to a score from 0 to 5 as follows:

# test_val = 100 * result
# score = math.ceil(17 - test_val)
# if score > 5:
#    score = 5

# print(score)
