import tensorflow as tf
import numpy as np

from keras import Sequential
from keras.api.layers import Dense

# A simple neural network with single layer and one neuron

# Just one dense layer with one neuron (unit)
model = Sequential([Dense(units=1, input_shape=[1])])
# sgd stands for Stochastic Gradient Descent. The optimizer makes guesses for Y = WX + B, containing weight and bias)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Y = 2*X - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Begin the learning process. Fit Xs to Ys, try it 500 times.
model.fit(xs, ys, epochs=500)

# Expect ~ (2*10 - 1) i.e.  ~19
print(model.predict(np.array([10.0])))

# Importance of Normalized data for training NNs:
# Note for larger value of X (e.g. 40.0 and 79.0), the training fails quickly. Giving [nan].