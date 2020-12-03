from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from matplotlib import pyplot
import time
import SWatGenerator

def train(trainFile, validateFile):
    generator1 = SWatGenerator.SWatGenerator(
        trainFile, "", 1000)
    generator2 = SWatGenerator.SWatGenerator(
        validateFile, "", 1000)
    generator2.normaliser.min_val = generator1.normaliser.min_val
    generator2.normaliser.max_val = generator1.normaliser.max_val
    generator2.normaliser.mean_val = generator1.normaliser.mean_val

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(100, 5)))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(5))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit_generator(generator=generator1, epochs=10, validation_data=generator2, verbose=2,
                        shuffle=False)
    return model, history

if __name__ == "__main__":
    train('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 'C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\validateChunk.csv')
