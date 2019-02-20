from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10


x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True,input_shape=(timesteps, data_dim)))  
 # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  
# return a single vector of dimension 32
model.add(LSTM(32))  

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=128, epochs=5,
          validation_data=(x_val, y_val))