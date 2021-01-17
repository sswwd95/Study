import numpy as np
import pandas as pd

sam = np.load('./samsung/npy/samsung3_data.npy',allow_pickle='True')
ko = np.load('./samsung/npy/ko_data.npy',allow_pickle='True')

print(sam.shape)
print(sam)
print(ko.shape)