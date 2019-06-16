# *************************************************************************
# ******************************************** Imports *********************************************
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
import matplotlib.pyplot as plot
import requests
import numpy as np


# ****************************************** Callback Class ****************************************
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        desired_accuracy = 0.90
        if logs.get('acc') > desired_accuracy:
            print("\nReached " + (desired_accuracy*100).__str__() + "% accuracy so cancelling training!")
            self.model.stop_training = True


# ****************************************** Get API Key *******************************************
def get_api_key():
    api_key = ""
    api_filename = "API_KEY.txt"
    api_string = "AlphaAdvantage"

    with open(api_filename, 'r') as f:
        line = f.readline()
        if api_string in line:
            api_key = line.split(":-")[1]
            f.close()

    return api_key


# ****************************************** Get API Data ******************************************
def get_api_data(api_key, symbol):
    data_list = []
    api_data = None
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

    # Iterate through all items in JSON file
    for item in api_data["Time Series (Daily)"]:
        item_dict = {
            "date": item,
            "open": float(api_data["Time Series (Daily)"][item]["1. open"]),
            "close": float(api_data["Time Series (Daily)"][item]["4. close"])
        }

        data_list.append(item_dict)

    # Reverse data_list to have order as old -> new
    data_list.reverse()

    return data_list


# ****************************************** Plot API Data *****************************************
def plot_api_data(data_list):
    # Initialize list variable for plotting
    closing_list = []

    # Iterate over the api data and collect closing values
    for item in data_list:
        closing_list.append(item["close"])

    # Plot closing data
    plot.plot(closing_list)
    plot.show()


# ************************************** Pre-process Train Data ************************************
def preprocess_train_data(data_list, days_to_label=5):
    # Initialize variables
    training_data = np.empty(shape=(1, days_to_label))
    training_labels = np.empty(shape=(1,), dtype=int)

    # Setup training data and labels
    for index, item in enumerate(data_list):
        # Get the data for current day
        if index > days_to_label:
            tomorrow = item
            index_train_data = np.array([])

            # Training data based on days_to_label
            for i in range(days_to_label):
                index_train_data = np.append(index_train_data, data_list[index - i]["close"])

            # Training label based on the next days performance (up = 1, down = 0)
            train_label = 1 if (tomorrow.get("open") < tomorrow.get("close")) else 0

            # Add training data and label to list
            training_data = np.vstack((training_data, index_train_data))
            training_labels = np.vstack((training_labels, train_label))
    training_labels = np.delete(training_labels, 0, 0)
    training_labels = np.squeeze(np.asarray(training_labels))
    training_data = np.delete(training_data, 0, 0)

    return [training_data, training_labels]

# *************************************** Normalize Train Data *************************************
def normalize_data_list(data_list):
    # Initialize variables
    open_max = 0
    close_max = 0

    # Iterate list and find maxes
    for item in data_list:
        temp_open = item["open"]
        temp_close = item["close"]

        if temp_open > open_max:
            open_max = temp_open
        if temp_close > close_max:
            close_max = temp_close

    # Iterate list and normalize data
    for item in data_list:
        item["open"] = item["open"] / open_max
        item["close"] = item["close"] / close_max

    return data_list


# **************************************** Model Definition ****************************************
def model_def(training_data):
    # Initialize variables
    model = tf.keras.models.Sequential()

    # Create NN layers
    batch_size, input_dim = training_data.shape
    in_neurons_1 = 10
    in_neurons_2 = 10
    out_neurons = 1

    model.add(tf.keras.layers.Dense(in_neurons_1, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.Dense(in_neurons_2, activation='relu'))
    model.add(tf.keras.layers.Dense(out_neurons, activation='sigmoid'))

    # Specify NN optimizing method and loss calculation method
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# ******************************************** Model Fit *******************************************
def model_fit(model, training_data, training_labels, epochs=100):
    # Fit training data
    model.fit(training_data, training_labels, epochs=epochs, callbacks=[MyCallback()],
              verbose=2, use_multiprocessing=True)

    return model


# ***************************************** Model Evaluate *****************************************
def model_eval(model, test_data, test_labels):
    # Fit test data and evaluate
    model.evaluate(test_data, test_labels)


# ***************************************** Main Function ******************************************
# Main for calling all functions in
def main():
    print("Starting program")
    data_list = get_api_data(get_api_key(), "AAPL")
    data_list = normalize_data_list(data_list)

    [training_data, training_labels] = preprocess_train_data(data_list, days_to_label=5)

    stock_model = model_def(training_data)
    stock_model = model_fit(stock_model, training_data, training_labels, epochs=50)


# ************************************* Main Function Execute **************************************
# Execute the main function
main()