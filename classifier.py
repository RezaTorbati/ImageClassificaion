import pickle
import numpy

file = "cifar-10-batches-py"
dict = 0
with open(file + "/data_batch_1", "rb") as fo:
    dict = pickle.load(fo, encoding='bytes')

print(type(dict))
for i in dict:
    print(i)

labels = dict[b'labels']
data = dict[b'data']
print(data)
print(type(data))