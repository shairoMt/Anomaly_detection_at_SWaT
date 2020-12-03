import pandas as pd
import numpy as np

class Normaliser(object):

    def __init__(self, filePath, batchSize):
        self.filePath = filePath
        self.batchSize = batchSize
        self.min_val, self.max_val, self.mean_val, self.length = self.findNormaliser()

    def findNormaliserOfBatch(self,  batch):
        length = np.shape(batch)[0]
        min_val = np.min(batch, axis=0)
        max_val = np.max(batch, axis=0)
        mean_val = np.mean(batch, axis=0)
        return length,min_val,max_val,mean_val

    def readAndFilterBatchFromCsv(self, start):
        batch = pd.read_csv(self.filePath,
                           sep=",",
                           decimal=".",
                           skiprows=start+1,
                           nrows=self.batchSize
        )
        batch = pd.DataFrame.to_numpy(batch)[:, 2:7]
        return batch

    def findNormaliser(self):
        start = 0
        i = 0
        output = self.readAndFilterBatchFromCsv(start)
        if(output.size == 0):
            return -1
        length, min_val, max_val, mean_val = self.findNormaliserOfBatch(output)
        while(True):
            start = start + length
            output = self.readAndFilterBatchFromCsv(start)
            if(output.size == 0):
                break
            length, min_val_, max_val_, mean_val_ = self.findNormaliserOfBatch(output)
            i = i+1
            for j in range(len(min_val)):
                if(min_val[j] > min_val_[j]):
                    min_val[j] = min_val_[j]
                if (max_val[j] < max_val_[j]):
                    max_val[j] = max_val_[j]
                mean_val[j] = (mean_val[j]*i*self.batchSize + mean_val_[j]*length)/(i * self.batchSize + length)
        length = length + self.batchSize * i
        return min_val, max_val, mean_val, length

    def normaliseBatch(self, batch):
        for i in range(len(self.mean_val)):
            batch[:,i] = batch[:,i] - self.mean_val[i]
            batch[:,i] = batch[:,i] / (self.max_val[i] - self.min_val[i])
        return batch

    def unnormalise(self, normalised):
        unnormalised = normalised * (self.max_val - self.min_val) + self.mean_val
        return np.array(unnormalised)

