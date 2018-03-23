"""
Module to train the models.
We just have to train two types of models:
1. ConvNet written from scratch
2. Transfer Learning and Fine Tuning
"""
import numpy as np
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train(model, x_train, y_train, split=0.8, epochs=1, batch_size=32, early_stopping=False):
    """
    Takes the traing data and its labels. Processes them and train the
    desired model. The data is normalized and randomly shuffled.

    Args:


    Returns:
        split: a decimal number between 0 and 1. This proportion will
                    be used for training the model
        model: the model to be trained
        x_train: numpy array which is the training data
        y_train: numpy array which is acts as the corresponding data
        epoch: number of epochs
        early_stopping: set True if early stopping is required
    """

    datagen = image.ImageDataGenerator(featurewise_center=False,
                                        featurewise_std_normalization=False)
    seed = np.arange(len(x_train))
    np.random.shuffle(seed)
    train_data = x_train[seed]
    train_labels = y_train[seed]

    train_size = int(split * len(x_train))

    x_val = x_train[train_size:]
    y_val = y_train[train_size:]

    x_train = train_data[:train_size]
    y_train = train_labels[:train_size]

    print('Training Model......')
    datagen.fit(x_train)

    if early_stopping is True:
        early_stopping_monitor = EarlyStopping(patience=2)
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) /batch_size,
                               epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping_monitor])
        return history, model
    else:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) /batch_size,
                               epochs=epochs, validation_data=(x_val, y_val))
        return history, model
