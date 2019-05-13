# *************************************************************************
# ******************************** Imports ********************************
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import requests
import numpy as np


# ************************** Variable Definitions *************************
global training_data, training_labels, test_data, test_labels, model, api_key, api_data


# ***************************** Pre-processing ****************************
# Check TensorFlow version
print(tf.__version__)


# ****************************** Callback Class ***************************
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        desired_accuracy = 0.90
        if (logs.get('acc') > desired_accuracy):
            print("\nReached " + (desired_accuracy*100).__str__() + "% accuracy so cancelling training!")
            self.model.stop_training = True


# ****************************** Get CSV Data *****************************
def get_api_key():
    # Global variables
    global api_key
    api_key = ""
    api_filename = "API_KEY.txt"
    api_string = "AlphaAdvantage"

    with open(api_filename, 'r') as f:
        line = f.readline()
        if api_string in line:
            api_key = line.split(":-")[1]
            f.close()


# ****************************** Get API Data *****************************
def get_api_data(symbol):
    # Global variables
    global api_key, api_data
    query = "https://www.alphavantage.co/query?"
    function_string = "function=TIME_SERIES_DAILY_ADJUSTED"
    symbol_string = "&symbol=" + symbol
    size_string = "&outputsize=full"
    api_string = "&apikey=" + api_key
    datatype_string = "&datatype=csv"

    total = query + function_string + symbol_string + size_string + api_string

    response = requests.get(total)

    if response.status_code == 200:
        api_data = response.json()
    else:
        print("API data loading failed.")


# ************************** Preprocess Train Data ************************
def preprocess_train_data():
    # Global variables
    global api_data, training_data, training_labels
    days_to_label = 3
    data_list = list()
    training_data = np.ndarray(shape=(1, days_to_label,))
    training_labels = np.ndarray(shape=(1,))

    open_max = 0
    close_max = 0

    # Iterate through all items in JSON file
    for item in api_data["Time Series (Daily)"]:
        item_dict = {
            "date": item,
            "open": float(api_data["Time Series (Daily)"][item]["1. open"]),
            "close": float(api_data["Time Series (Daily)"][item]["4. close"])
        }

        if (open_max < item_dict["open"]):
            open_max = item_dict["open"]
        if (close_max < item_dict["close"]):
            close_max = item_dict["close"]

        data_list.append(item_dict)

    # Normalize data
    for item in data_list:
        item["open"] = item["open"]/open_max
        item["close"] = item["close"] / close_max

    # Setup training data and labels
    for index, item in enumerate(data_list):
        # Get the data for "today"
        if (index < len(data_list) - days_to_label):
            next_day = item
            today = data_list[index + 1]
            yesterday = data_list[index + 2]
            yesterday_minus_1 = data_list[index + 3]

            # Training data based on day, day-1, day-2 close data
            train_data = np.array([
                today.get("close"),
                yesterday.get("close"),
                yesterday_minus_1.get("close")
            ])

            # Training label based on the next days performance (up = 1, down = 0)
            train_label = 1 if (next_day.get("open") < next_day.get("close")) else 0

            training_data = np.vstack((training_data, train_data))
            training_labels = np.vstack((training_labels, train_label))
    training_labels = np.delete(training_labels, 0, 0)
    training_data = np.delete(training_data, 0, 0)


# **************************** Model Definition ***************************
def model_def():
    # Global variables
    global training_data, model

    # Create NN layers
    batch_size, input_dim = training_data.shape
    hidden_layer_nodes = 10
    output_layer_nodes = 1
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(hidden_layer_nodes, input_dim=input_dim,
                                                              kernel_initializer='normal', activation='relu'),
                                        tf.keras.layers.Dense(output_layer_nodes, kernel_initializer='normal',
                                                              activation='sigmoid')])

    # Specify NN optimizing method and loss calculation method
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


# ******************************** Model Fit ******************************
def model_fit():
    # Global variables
    global training_data, training_labels, model

    # Fit training data
    model.fit(training_data, training_labels, epochs=15, callbacks=[MyCallback()], verbose=2)


# ***************************** Model Evaluate ****************************
def model_eval():
    # Global variables
    global training_data, training_labels, test_data, test_labels, model

    # Fit test data and evaluate
    model.evaluate(test_data, test_labels)


# ***************************** Main Function *****************************
# Main for calling all functions in
def main():
    print("Starting program")
    get_api_key()
    get_api_data("AAPL")
    preprocess_train_data()

    #model_def()
    #model_fit()


# ************************* Main Function Execute *************************
# Execute the main function
main()