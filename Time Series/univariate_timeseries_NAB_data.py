import tensorflow as tf
import csv
import numpy as np
from tensorflow import keras

def raw_data_to_input(data_source):
    # Initialize lists
    time_step = []
    nab_data = []
    # Open CSV file
    with open(data_source) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        # Append row and sunspot number to lists
        for row in reader:
            time_step.append(int(row[0]))
            nab_data.append(float(row[2]))
    # Convert lists to numpy arrays
    time = np.array(time_step)
    series = np.array(nab_data)
    return series, time

def split_train_val_test(series):
    n = len(series)
    train_df = series[0:int(n*0.90)]
    val_df = series[int(n*0.90):]
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    return train_df, val_df, train_mean, train_std

def windowed_dataset(series, window_size, batch_size, shuffle_buffer=False):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def data_split_transform(series, train_window_size, train_batch_size, val_window_size, val_batch_size, train_shuffle):
    train_series, val_series, train_mean, train_std = split_train_val_test(series)
    train_dataset = windowed_dataset(train_series, train_window_size, train_batch_size, train_shuffle)
    val_dataset = windowed_dataset(val_series, val_window_size, val_batch_size, False)
    return train_dataset, val_dataset, train_mean, train_std

def model_build():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1),
            ])
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.0008, momentum=0.7)
    model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
    return model

def model_training(train_series, val_series, patience, checkpoint_filepath):
    model = model_build()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', mode='min', save_best_only=True)
    history = model.fit(train_series, epochs=Epochs, callbacks=[early_stopping, model_checkpoint],
                        validation_data=val_series, verbose=1)
    return history
#Here in the exam if you see history/return history then go with that.


train_window_size = 60
train_batch_size = 32
train_shuffle = 1000
val_window_size = 32
val_batch_size = 16
patience = 30
Epochs = 100

checkpoint_filepath = "/Users/chidam_sp/PycharmProjects/pythonProject2/Computer vision, Time series, and NLP_TF certification/Numenta_Anomaly_Benchmark_(NAB)/LSTM_Dense_model.h5"

input_data_filepath = "/Users/chidam_sp/PycharmProjects/pythonProject2/Computer vision, Time series, and NLP_TF certification/Numenta_Anomaly_Benchmark_(NAB)/df_small_noise.csv"

series, time = raw_data_to_input(input_data_filepath)

train_series, val_series, train_mean, train_std = data_split_transform(series, train_window_size, train_batch_size, val_window_size, val_batch_size, train_shuffle)

history = model_training(train_series, val_series, patience, checkpoint_filepath)

# If needed - ancillary function to dump the training history as pickle file
# def download_history():
#   import pickle
#   with open('history_univariate_time_series.pkl', 'wb') as f:
#     pickle.dump(history.history, f)
#   with open('history_univariate_time_series.pkl', 'rb') as handle:
#     download_history = pickle.load(handle)
#   return download_history
#
# download_history()

# If you have to load a pickle file using a filepath
# import pickle
# with open('/Users/chidam_sp/PycharmProjects/pythonProject2/Computer vision, Time series, and NLP_TF certification/Numenta_Anomaly_Benchmark_(NAB)/history_univariate_time_series.pkl', 'rb') as handle:
#   b = pickle.load(handle)
# print(b)

#Load the final saved model
final_model = tf.keras.models.load_model(checkpoint_filepath)
final_model.summary()
