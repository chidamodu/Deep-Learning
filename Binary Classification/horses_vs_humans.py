import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from shutil import copyfile
import zipfile
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=45,
                                       width_shift_range=0.2,
                                       height_shift_range=0.3,
                                       zoom_range=[.8, 1],
                                       channel_shift_range=20,
                                       shear_range=0.4,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=128, class_mode='binary', target_size=(300, 300))

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=32, class_mode='binary',
                                                                  target_size=(300, 300))

    return train_generator, validation_generator


def model_build():
    # data_augmentation = tf.keras.Sequential([
    #     layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.2),
    # ])
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.93):
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training = True


def model_training(TRAINING_DIR, VALIDATION_DIR, EPOCHS):
    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)
    callbacks = myCallback()
    model = model_build()
    final_model = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, verbose=2,
                        callbacks=[callbacks])
    return final_model


TRAINING_DIR = "//horses_vs_humans/training/"
VALIDATION_DIR = "//horses_vs_humans/validation/"

final_model = model_training(TRAINING_DIR, VALIDATION_DIR, EPOCHS=20)

# if necessary use the below code, but dont forget to transform the test images the same way you did the training and validation images
# eval_result = final_model.evaluate(TESTING_DIR)
# print("[test loss, test accuracy]:", eval_result)

final_model.save('Users/chidam_sp/PycharmProjects/pythonProject2/Computer vision, Time series, and NLP_TF certification/horses_vs_humans/final_trained_model_horses_vs_humans.h5')
print("Final model saved!")

----------------------------------------------------------------------------------------------------------------------------------------------------------------

# If you are using transfer learning then follow the below code
def create_pre_trained_model(local_weights_file):
  pre_trained_model = InceptionV3(input_shape = (300, 300, 3),
                                  include_top = False,
                                  weights = None)
  pre_trained_model.load_weights(local_weights_file)
  # Make all the layers in the pre-trained model non-trainable
  for layer in pre_trained_model.layers:
    layer.trainable = False
  return pre_trained_model


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=-45,
                                       width_shift_range=-0.2,
                                       height_shift_range=0.3,
                                       # brightness_range = (0.2, 0.45),
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=128, class_mode='binary', target_size=(300, 300))

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=32, class_mode='binary',
                                                                  target_size=(300, 300))

    return train_generator, validation_generator


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.93):
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training = True


def output_of_last_layer(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_desired_layer.output
    return last_output


def create_final_model(pre_trained_model, last_output):
    from tensorflow.keras import Model
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=pre_trained_model.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def model_training_pretrained(EPOCHS, pre_trained_model, TRAINING_DIR, VALIDATION_DIR):

    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

    callbacks = myCallback()
    last_output = output_of_last_layer(pre_trained_model)
    model = create_final_model(pre_trained_model, last_output)
    final_model = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, verbose=2,
                        callbacks=[callbacks])
    return final_model

TRAINING_DIR = "//horses_vs_humans/training/"
VALIDATION_DIR = "//horses_vs_humans/validation/"
local_weights_file = '//horses_vs_humans/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = create_pre_trained_model(local_weights_file)
final_model_pretrained = model_training_pretrained(20, pre_trained_model, TRAINING_DIR, VALIDATION_DIR)

final_model_pretrained.save('Users/chidam_sp/PycharmProjects/pythonProject2/Computer vision, Time series, and NLP_TF certification/horses_vs_humans/final_trained_model_horses_vs_humans_using_pretrained.h5')
print("Final model saved!")