import string
import numpy as np
import pandas as pd
from numpy import array
from keras import optimizers
import matplotlib.pyplot as plt
from dateutil.parser import parse
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#creating a sequence
length = 100
seq = array([i / 100 for i in range(length)])
print(seq)

X = seq[:-1]
X = X.reshape(len(X), 1, 1)
y = seq[1:]
y = y.reshape(len(y), 1)


n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(1000, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=32,
          validation_split=0.2, shuffle=True)

#predict
model.predict(X[0:10])
