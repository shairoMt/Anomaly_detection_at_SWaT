import numpy as np
import Normaliser



def test_readAndFilterBatchFromCsv():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 10)
    x = n.readAndFilterBatchFromCsv(0)
    Erwartet = np.array([0 ,124.3135, 1, 1, 2])
    np.testing.assert_array_almost_equal(x[0], Erwartet)
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 10)
    x = n.readAndFilterBatchFromCsv(998)
    Erwartet = np.array([ 2.501041, 253.8476, 2, 2, 1])
    np.testing.assert_array_almost_equal(x[0], Erwartet)

def test_equality_findNormaliserOfBatch():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 3)
    output = n.readAndFilterBatchFromCsv(996)
    length,min_val,max_val,mean_val = n.findNormaliserOfBatch(output)
    min_val = np.array(min_val)
    max_val = np.array(max_val)
    mean_val = np.array(mean_val)
    np.testing.assert_equal(length,3)
    np.testing.assert_array_almost_equal(min_val,[2.436025,253.4551,2,2,1])
    np.testing.assert_array_almost_equal(max_val,[2.501041,253.8476,2,2,1])
    np.testing.assert_array_almost_equal(mean_val,[2.4648496667,253.58593333,2,2,1])

def test_not_equality_findNormaliserOfBatch():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 3)
    output = n.readAndFilterBatchFromCsv(0)
    length,min_val,max_val,mean_val = n.findNormaliserOfBatch(output)
    min_val = np.array(min_val)
    max_val = np.array(max_val)
    mean_val = np.array(mean_val)
    np.testing.assert_equal(length,3)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, min_val, [2.436025,253.4551,2,2,1])
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, max_val, [2.501041,253.8476,2,2,1])
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, mean_val, [2.4648496667,253.58593333,2,2,1])

def test_findNormaliser():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 1000)
    min_val = n.min_val
    max_val = n.max_val
    mean_val = n.mean_val

    output = n.readAndFilterBatchFromCsv(0)
    min_val_ = np.min(output, axis=0)
    max_val_ = np.max(output, axis=0)
    mean_val_ = np.mean(output, axis=0)
    np.testing.assert_array_almost_equal(min_val, min_val_)
    np.testing.assert_array_almost_equal(max_val,max_val_)
    np.testing.assert_array_almost_equal(mean_val,mean_val_)

def test_normaliseBatch():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 3 )
    batch = n.readAndFilterBatchFromCsv(0)
    normalised_batch = n.normaliseBatch(batch)
    np.testing.assert_array_equal(normalised_batch[0],[-0.3710788083867222, -0.22707547911555165, -0.1905,
       -0.1259999999999999, 0.9989999999999999])

def test_unnormalise():
    n = Normaliser.Normaliser('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', 1 )
    batch = n.readAndFilterBatchFromCsv(0)
    normalisedBatch = n.normaliseBatch(batch)
    unnormalised = n.unnormalise(normalisedBatch)
    np.testing.assert_equal(np.array([0.0, 124.3135, 1.0, 1.0, 2.0]),unnormalised[0] )


if __name__ == "__main__":
    test_readAndFilterBatchFromCsv()
    test_equality_findNormaliserOfBatch()
    test_not_equality_findNormaliserOfBatch()
    test_findNormaliser()
    test_normaliseBatch()
    test_unnormalise()
