import SWatGenerator
import numpy as np

def test_len():
    generator = SWatGenerator.SWatGenerator('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', "", 100)
    np.testing.assert_equal(len(generator), 9)

def test_getitem():
    generator = SWatGenerator.SWatGenerator('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\py project\\chunk.csv', "", 3)
    samples, labels = generator[0]
    np.testing.assert_equal(samples.shape, (3,100,5))
    np.testing.assert_equal(labels.shape, (3,5))
    np.testing.assert_equal(samples[0][0], np.array([0, 124.3135, 1, 1, 2]))

if __name__ == "__main__":
    test_len()
    test_getitem()