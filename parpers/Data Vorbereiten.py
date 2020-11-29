import pandas as pd
import numpy as np



def readAndFilterBatchFromCsv(filePath ,start ,batchSize ):
    output = pd.read_csv(filePath,
                       sep=",",
                       decimal=".",
                       skiprows=start+1,
                       nrows=batchSize+1
    )

    if(output.size < (batchSize-start)):
        output = pd.DataFrame.to_numpy(output)[:, 2:7]
        return output

    output = pd.DataFrame.to_numpy(output)[:, 2:7]
    return output


x = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv',997,10000)
print(x)



def test1():
    x = readAndFilterBatchFromCsv('C:\\Users\\kaspe\\Desktop\\Anomaly_detection_at_SWaT\\chunk.csv',997,10000)
    Erwartet = np.array([[2.457483,253.4551,2,2,1],[2.501041,253.8476,2,2,1]])
    return np.testing.assert_almost_equal(x, Erwartet)

