import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def load_mnist():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels, test_images, test_labels

def preprocessing_data(images):
    images = images.reshape((images.shape[0], 28, 28, 1))
    images = images / 255.0
    return images

def model_build():
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.247),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.00003, momentum=0.356),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'final_trained_model.h5', monitor='val_loss',mode='min', save_best_only=True)#only if necessary
# # Instantiate the callback class
# callbacks = myCallback()

train_images, train_labels, test_images, test_labels = load_mnist()
train_images = preprocessing_data(train_images)
test_images = preprocessing_data(test_images)

def model_train(train_images, train_labels):
    model = model_build()
    # Split the training data into training and validation sets
    train_images_split, val_images_split, train_labels_split, val_labels_split = \
        (train_test_split(train_images, train_labels, test_size=0.2, random_state=2078))
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model.fit(
        train_images_split, train_labels_split,
        batch_size=256,
        epochs=20,
        verbose=1,
        validation_data=(val_images_split, val_labels_split), callbacks=[stop_early]
    )
    return model

final_model = model_train(train_images, train_labels)

eval_result = final_model.evaluate(test_images, test_labels)
print("[test loss, test accuracy]:", eval_result)

final_model.save('/Users/chidam_sp/PycharmProjects/pythonProject2/intro_TF_week3/final_trained_model_for_py_format.h5')
print("Final model saved!")



