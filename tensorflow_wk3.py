
# *************************************************************************
# ******************************** Imports ********************************
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ************************** Variable Definitions *************************
global mnist, training_images, training_labels, test_images, test_labels, model

# ***************************** Pre-processing ****************************
# Check TensorFlow version
print(tf.__version__)

# **************************** Load MNIST DATA ****************************
# Load mnist data into variables
def load_MNIST_data():
    # Global variables
    global mnist, training_images, training_labels, test_images, test_labels, model

    # Load MNIST fashion data
    mnist = tf.keras.datasets.fashion_mnist

    # Assign training and test data and labels
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape and normalize all training and test images, the reshape makes a 4D matrix (60000x28x28x1) as that is what
    # the convolution operations expect as input
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

# *********************** MNIST Fashion Convolution ***********************
# Labeled data set of 28x28 pixel images of clothing items
# Using a 3 layer NN and 2 layers of convolution, train data and test with test data
def MNIST_fashion_convo():
    # Global variables
    global mnist, training_images, training_labels, test_images, test_labels, model

    # Create NN model by setting layers of convolution and the actual NN model, this takes several minutes
    # - Line 1 is for setting a 3x3 convolution matrix and setting the input shape to MNIST images 28x28
    # - Line 2 is for setting a 2x2 pooling matrix, a pixel and its neighbors (right, bottom, and b-r) are pooled into 1
    # - Line 3 is for setting another 3x3 convolution matrix
    # - Line 4 is for setting another 2x2 pooling matrix
    # - Lines 5-7 are the shape of our NN as used before
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Assign optimizer and loss calculations, compile and fit data, summary adds detail to our NN
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=5, verbose=2)

    # Evaluate with test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)


# ************************** Visualizing Pooling **************************
# This next example will allow us to visualize convolutions and pooling
def visualize_pooling():
    # Global variables
    global mnist, training_images, training_labels, test_images, test_labels, model

    f, axarr = plt.subplots(3,4)
    FIRST_IMAGE=0
    SECOND_IMAGE=7
    THIRD_IMAGE=26
    CONVOLUTION_NUMBER = 1
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for x in range(0,4):
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0,x].grid(False)
        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1,x].grid(False)
        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2,x].grid(False)


# ***************************** Main Function *****************************
# Main for calling all functions in
def main():
    load_MNIST_data()
    MNIST_fashion_convo()
    visualize_pooling()

# ************************* Main Function Execute *************************
# Execute the main function
main()

