import random
import numpy as np
from sklearn.metrics import confusion_matrix
from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco
from Utils.reshape_data import reshape


def sigmoid(z):
    try:
        s = 1 / (1 + np.exp(-z))
    except RuntimeWarning:
        s = 0
    return s


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    try:
        s = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    except RuntimeWarning:
        s = 0
    return s


def neural_network(X, Y, x_test=None, y_test=None, activation_function='sigmoid'):
    learning_rate = .5

    # Initalize weights to small number between -.05 and .05
    w = np.array([random.uniform(-.05, .05) for _ in range(len(X[0]))])
    b = 0

    for i in range(1000):
        m = X.shape[1]

        # Forward Propagate
        A = []
        if activation_function == 'sigmoid':
            A = sigmoid(np.dot(X, w) + b)
        elif activation_function == 'relu':
            A = relu(np.dot(X, w) + b)
        elif activation_function == 'tanh':
            A = tanh(np.dot(X, w) + b)

        # Backward propagate
        dw = (1 / m) * np.dot((A - Y).T, X)
        db = (1/m) * np.sum(A-Y)

        # update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 1000 == 0:
            print("loop:", i)

    pred = predict(w, b, X)
    print("Confusion Matrix =\n", confusion_matrix(Y, pred))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100))

    if x_test is not None:
        pred = predict(w, b, x_test)
        print("Confusion Matrix =\n", confusion_matrix(y_test, pred))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(y_test))) * 100))


def predict(w, b, X, activation_function='sigmoid'):
    m = X.shape[0]
    pred = np.zeros(m)

    w = w.reshape(X.shape[1], 1)

    A = []
    if activation_function == 'sigmoid':
        A = sigmoid(np.dot(X, w) + b)
    elif activation_function == 'relu':
        A = relu(np.dot(X, w) + b)
    elif activation_function == 'tanh':
        A = tanh(np.dot(X, w) + b)

    for i in range(A.shape[0]):
        pred[i] = 1 if A[i] > .5 else 0
    return pred


if __name__ == "__main__":
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
    neural_network(x, y)

    # # Run on aruco
    # train_x, train_y, test_x, test_y = load_dataset_aruco(1000)
    # neural_network(train_x, train_y, test_x, test_y, 'sigmoid')

    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset(100)
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    neural_network(train_x, train_y, test_x, test_y, 'sigmoid')

