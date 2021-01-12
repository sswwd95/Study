import numpy as np
a = np.array(range(1,11))
size = 5

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

x = dataset[:,:4]
y = dataset[:,4]

print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0],x.shape[1],1)

from tensorflow.keras.models import load_model
model = load_model("../data/h5/save_keras35.h5")

model.summary()

# WARNING:tensorflow:No training configuration found 
# in the save file, so the model was *not* compiled. Compile it manually.
# 위의 문구는 가중치 저장을 안해서 뜨는 것

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=20, mode='min')

model.fit(x, y, batch_size=1, callbacks=[early_stopping],epochs=1000)

#4. 평가,예측
loss = model.evaluate(x,y, batch_size=1)
print('loss : ',loss)

x_pred = np.array([7,8,9,10])
x_pred = x_pred.reshape(1,4,1)

result = model.predict(x_pred)
print('result : ',result)

# loss :  0.0750507190823555
# result :  [[10.734548]]