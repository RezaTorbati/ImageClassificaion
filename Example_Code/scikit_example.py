import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, log_loss
from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco, load_dataset_full
from Utils.reshape_data import reshape
import time

train_x, train_y, test_x, test_y = load_dataset()
train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)

tic = time.perf_counter()
print("Training Network")
neural_network = MLPClassifier(hidden_layer_sizes=(20, 7, 5))
neural_network.fit(train_x, np.ravel(train_y))

print("Took:", time.perf_counter() - tic, " seconds to train")

print("Train")
pred = neural_network.predict(train_x)
print("Confusion Matrix =\n", confusion_matrix(train_y, pred))
print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(train_y))) * 100))

print("Test")
pred = neural_network.predict(test_x)
print("Confusion Matrix =\n", confusion_matrix(test_y, pred))
print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(test_y))) * 100))

