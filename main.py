import numpy as np
from Neural_Network.neural_network import NeuralNetwork, Layer
from Neural_Network.non_binary_perceptron_array import nb_perceptron
from Neural_Network.perceptron import perceptron
from sklearn.metrics import confusion_matrix
from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco, load_dataset_full
from Utils.reshape_data import reshape
import time

from arucoLRClassification import aruco_lr
from cifarLRClassification import cifar_lr


def run_nn_cifar():
    print("\nRunning Neural Network on CIFAR-10")
    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    neural_network = NeuralNetwork(500)
    neural_network.fit(train_x, train_y, [Layer(20, len(train_x[0])), Layer(20, 20), Layer(1, 20)])

    prediction = neural_network.predict(train_x)
    print("Confusion Matrix =\n", confusion_matrix(train_y, prediction))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(train_y))) * 100))

    prediction = neural_network.predict(test_x)
    print("Confusion Matrix =\n", confusion_matrix(test_y, prediction))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(test_y))) * 100))


def run_nn_aruco():
    print("\nRunning Neural Network on ARUCO")
    # Run on ARUCO
    train_x, train_y, test_x, test_y = load_dataset_aruco()
    neural_network = NeuralNetwork(500)
    neural_network.fit(train_x, train_y, [Layer(20, len(train_x[0])), Layer(1, 20)])

    prediction = neural_network.predict(train_x)
    print("Confusion Matrix =\n", confusion_matrix(train_y, prediction))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(train_y))) * 100))

    prediction = neural_network.predict(test_x)
    print("Confusion Matrix =\n", confusion_matrix(test_y, prediction))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(test_y))) * 100))


def run_nn_test():
    # Trivial Example for testing
    x = np.array([[1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, 1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, 1, -1, -1, -1, -1, -1],
                  [-1, -1, -1, 1, -1, -1, -1, -1],
                  [-1, -1, -1, -1, 1, -1, -1, -1],
                  [-1, -1, -1, -1, -1, 1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, 1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, 1],
                  [-1, -1, -1, -1, -1, -1, -1, -1]])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0])
    neural_network = NeuralNetwork()
    neural_network.fit(x, y, [Layer(20, len(x[0])), Layer(7, 20), Layer(5, 7), Layer(1, 5)])

    prediction = neural_network.predict(x)
    print("Confusion Matrix =\n", confusion_matrix(y, prediction))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(y))) * 100))


def run_perceptrion():
    # Trivial Example for testing
    x = np.array([[1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, 1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, 1, -1, -1, -1, -1, -1],
                  [-1, -1, -1, 1, -1, -1, -1, -1],
                  [-1, -1, -1, -1, 1, -1, -1, -1],
                  [-1, -1, -1, -1, -1, 1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, 1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, 1],
                  [-1, -1, -1, -1, -1, -1, -1, -1]])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0])
    perceptron(x, y)

    print("\nRunning Perceptron on CIFAR-10")
    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset()
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    perceptron(train_x, train_y, test_x, test_y, 'sigmoid')

    print("\nRunning Perceptron on ARUCO")
    # Run on aruco
    train_x, train_y, test_x, test_y = load_dataset_aruco()
    perceptron(train_x, train_y, test_x, test_y, 'sigmoid')


def run_nb_perceptron():
    print("\nRunning Non-Binary Perceptron on CIFAR-10")
    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset_full()
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    nb_perceptron(train_x, train_y, test_x, test_y, 'sigmoid', num_interations=1000)


if __name__ == "__main__":
    print("\nRunning logistic regression on CIFAR-10")
    cifar_lr()
    print("\nRunning logistic regression on ARUCO")
    aruco_lr()
    run_perceptrion()
    run_nb_perceptron()
    run_nn_test()
    run_nn_cifar()
    run_nn_aruco()







