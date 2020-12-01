import pandas as pd
import numpy as np


def findNormaliserOfBatch(batch):
    length = np.shape(batch)[0]
    min_val = []
    max_val = []
    mean_val = []
    for i in range(np.shape(batch)[1]):
        min_val.append(np.amin(batch[:, i]))
        max_val.append(np.amax(batch[:, i]))
        mean_val.append(sum(batch[:, i]) / length)
    return length,min_val,max_val,mean_val

def readAndFilterBatchFromCsv(filePath ,start ,batchSize ):
    batch = pd.read_csv(filePath,
                       sep=",",
                       decimal=".",
                       skiprows=start+1,
                       nrows=batchSize+1
    )

    if(batch.size < (batchSize-start)):
        batch = pd.DataFrame.to_numpy(batch)[:, 2:7]
        return batch

    batch = pd.DataFrame.to_numpy(batch)[:, 2:7]
    return batch

def findNormaliser(filePath):
    start = 0
    batchSize = 999
    min_val=[0,0,0,0,0]
    max_val=[0,0,0,0,0]
    mean_val=[0,0,0,0,0]
    length = 0
    min_val = []
    max_val = []
    mean_val = []
    i = 0

    output = readAndFilterBatchFromCsv(filePath, start, batchSize)
    if(output.size == 0):
        return -1
    length, min_val, max_val, mean_val = findNormaliserOfBatch(output)

    while(True):
        start = start + batchSize
        batchSize = batchSize + 1000
        output = readAndFilterBatchFromCsv(filePath,start,batchSize)

        if(output.size == 0):
            break
        length_,min_val_,max_val_,mean_val_ = findNormaliserOfBatch(output)
        for j in range(len(min_val)):
            if(min_val[j] > np.amin(output[:, j])):
                min_val[j] = np.amin(output[:, j])

            if (max_val[j] < np.amax(output[:, j])):
                max_val[j] = np.amax(output[:,j])

            mean_val[j] = (mean_val[j]*i*batchSize + mean_val_[j]*length)/(i * batchSize + length)
        i = i+1

    return min_val , max_val , mean_val

def normaliseBatch(batch ,min_val ,max_val ,mean_val):

    for i in range(len(mean_val)):
        batch[:,i] = batch[:,i] - mean_val[i]
        batch[:,i] = batch[:,i] / (min_val[i] + max_val[i])
    return batch



def test_readAndFilterBatchFromCsv():
    x = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv',997,10000)
    Erwartet = np.array([[2.457483,253.4551,2,2,1],[2.501041,253.8476,2,2,1]])
    return np.testing.assert_array_almost_equal(x, Erwartet)

def test_equality_findNormaliserOfBatch():
    x = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv',997,10000)
    length,min_val,max_val,mean_val = findNormaliserOfBatch(x)
    min_val = np.array(min_val)
    max_val = np.array(max_val)
    mean_val = np.array(mean_val)
    np.testing.assert_almost_equal(length,2)
    np.testing.assert_array_almost_equal(min_val,[2.457483,253.4551,2,2,1])
    np.testing.assert_array_almost_equal(max_val,[2.501041,253.8476,2,2,1])
    np.testing.assert_array_almost_equal(mean_val,[2.479262,253.65135,2,2,1])

def test_not_equality_findNormaliserOfBatch():
    x = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv', 997, 10000)
    length, min_val, max_val, mean_val = findNormaliserOfBatch(x)
    min_val = np.array(min_val)

    #x = np.array([  2.457483, 253.4551   ,  2 ,        2   ,      1      ])

    x = np.array([999999, 999999, 999999, 999999, 999999])
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, min_val, x)

def test_findNormaliser():
    min_val, max_val, mean_val = findNormaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv')
    output = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv' ,0 ,999)
    min_val_ = []
    max_val_ = []
    mean_val_ = []

    for i in range(len(output[0])):
        min_val_.append(np.amin(output[:,i]))
        max_val_.append(np.amax(output[:, i]))
        mean_val_.append(sum(output[:, i]) / len(output[:,0]))

    np.testing.assert_array_almost_equal(min_val, min_val_)
    np.testing.assert_array_almost_equal(max_val,max_val_)
    np.testing.assert_array_almost_equal(mean_val,mean_val_)


def test_normaliseBatch():
    batch = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv' ,0 ,3 )
    min_val ,max_val ,mean_val = findNormaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv')
    normalised_batch = normaliseBatch(batch ,min_val ,max_val ,mean_val)
    for i in range(len(mean_val)):
        batch[:,i] = batch[:,i] - mean_val[i]
        batch[:,i] = batch[:,i] / (min_val[i] + max_val[i])

    np.testing.assert_array_almost_equal(normalised_batch,batch)



test_readAndFilterBatchFromCsv()
test_equality_findNormaliserOfBatch()
test_not_equality_findNormaliserOfBatch()
test_findNormaliser()
test_normaliseBatch()
