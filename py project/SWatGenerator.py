import keras
import Normaliser
import numpy as np
class SWatGenerator(keras.utils.Sequence):

    def __init__(self, filename, labels, batch_size):
        self.filename = filename
        self.labels = labels
        self.batch_size = batch_size
        self.normaliser = Normaliser.Normaliser(self.filename, self.batch_size + 100)


    def __len__(self):
        return (int)((self.normaliser.length-100)/self.batch_size)

    def __getitem__(self, ind):
        output = self.normaliser.readAndFilterBatchFromCsv(self.batch_size * ind)
        samples = []
        labels = []
        for i in range(self.batch_size):
            samples.append(output[i:i+100, :])
            labels.append(output[i+100])
        return np.array(samples), np.array(labels)