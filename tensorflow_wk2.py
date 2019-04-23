
# ******************************** Imports ********************************
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ***************************** Pre-processing ****************************

# Check TensorFlow version
print(tf.__version__)

# Implement callback class for ending training based on criteria
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    desired_accuracy = 0.99
    if(logs.get('acc') > desired_accuracy):
      print("\nReached " + (desired_accuracy*100).__str__() + "% accuracy so cancelling training!")
      self.model.stop_training = True


# ************************ MNIST Fashion Example 1 ************************
# Labeled data set of 28x28 pixel images of clothing items
# Using a 3 layer NN, train data and test with test data

# Retrieve and load data into training and test data sets
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Plot data to understand data set better
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# Normalize data, whole arrays can be manipulated with 1 line
training_images = training_images / 255.0
test_images = test_images / 255.0

# Create NN layers, the .Flatter() call in the first layer takes our 2D array and "flattens" to a 1D vector
hidden_layer_nodes = 128
output_layer_nodes = 10
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(hidden_layer_nodes, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(output_layer_nodes, activation=tf.nn.softmax)])

# Specify NN optimizing method and loss calculation method
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit training data
callbacks = myCallback()
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Fit test data and evaluate
model.evaluate(test_images, test_labels)


# ********************** MNIST Handwriting Example 1 **********************
# Labeled data set of 28x28 pixel images of handwritten digits 0-9
# Using a 3 layer NN, train data and test with test data

# Load handwriting data set
mnist_handwrite = tf.keras.datasets.mnist

# Initialize training and test data sets
(x_train, y_train), (x_test, y_test) = mnist_handwrite.load_data()

# Normalize data, whole arrays can be manipulated with 1 line
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create NN layers, the .Flatter() call in the first layer takes our 2D array and "flattens" to a 1D vector
hidden_layer_nodes = 128
output_layer_nodes = 10
handwrite_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(hidden_layer_nodes, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(output_layer_nodes, activation=tf.nn.softmax)])

# Specify NN optimizing method and loss calculation method
handwrite_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit training data
callbacks = myCallback()
handwrite_model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

# Fit test data and evaluate
handwrite_model.evaluate(x_test, y_test)


