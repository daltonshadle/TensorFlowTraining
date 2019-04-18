
# ******************************** Imports ********************************
import tensorflow as tf
from tensorflow import keras
import numpy as np


# ************************ Linear Equation Example ************************
# Imagine the linear equation y = 2x -1
# Predict the solution if x = 10
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)
print("Expected value around 19. Expected value = ")
print(model.predict([10.0]))


# ************************* Housing Price Example *************************
# Imagine each house is 50k + 50k per bedroom
# Predict the price of a house with 7 bedrooms
house_model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
house_model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)

house_model.fit(xs, ys, epochs=500)
print("Expected value around 4. Expected value (in hundreds of thousands of $) = ")
print(house_model.predict([7.0]))


