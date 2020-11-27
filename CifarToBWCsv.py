import pickle
import numpy as np
import pandas as pd

directory = "cifar-10-batches-py"
dict = 0
files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
dfs = []
for f in files:
    file = directory + "/" + f
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding='bytes')

    labels = dict[b'labels']
    data = dict[b'data']

    bw = np.ones((10000, 1024))
    for i in range(0,len(data)):
        dataLength = int(len(data[i]) / 3)
        for j in range(0,dataLength):
            bw[i, j] = round((data[i, j] + data[i, j + dataLength] + data[i, j + dataLength * 2]) / 3) #gets the black and white value

    df = pd.DataFrame(data=bw)
    df['label'] = labels

    writeName = file + ".csv"
    df.to_csv(writeName, index = False)

    if(f != "test_batch"):
        dfs.append(df)

df = pd.concat(dfs)
writeName = directory + "/" + "data_batch_full.csv"
df.to_csv(writeName, index=False)