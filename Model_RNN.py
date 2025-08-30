import numpy as np
from keras.src.layers import LSTM, Dense
# https://www.tensorflow.org/guide/keras/rnn
# from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from keras.layers import Conv1D
import tensorflow as tf

def Model_RNN(train_data, train_target, test_data, test_target, sol=50):
    pred, model = RNN_train(train_data, train_target, test_data, sol)  # RNN
    pred = np.squeeze(pred)

    Eval = (pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model


def Model_MRNN_AM(train_data, train_target, test_data, test_target, sol=50):
    pred, model = MRNNAM_train(train_data, train_target, test_data, sol)  # RNN
    pred = np.squeeze(pred)

    Eval = (pred, test_target)
    return Eval, pred


def MRNNAM_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    conv1 = Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))
    conv2 = Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))
    conv3 = Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))
    conv = (conv1+conv2+conv3)/3
    model.add(conv)
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model