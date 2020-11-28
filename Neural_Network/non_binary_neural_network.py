# TODO:
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from Utils.load_cifar10_keras import load_dataset, load_dataset_full
from Utils.reshape_data import reshape
from pprint import pprint

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


def neural_network(X, Y, x_test=None, y_test=None, activation_function='sigmoid', num_interations=1000):
    learning_rate = .01

    # Initalize weights to small number between -.05 and .05
    w = [np.array([random.uniform(-.05, .05) for _ in range(len(X[0]))]) for _ in range(len(set(Y)))]
    b = 0

    for i in range(num_interations):
        for j, output in enumerate(w):
            m = X.shape[1]

            # Forward Propagate
            A = []
            if activation_function == 'sigmoid':
                A = sigmoid(np.dot(X, w[j]) + b)
            elif activation_function == 'relu':
                A = relu(np.dot(X, w[j]) + b)
            elif activation_function == 'tanh':
                A = tanh(np.dot(X, w[j]) + b)

            # Backward propagate
            # Select the values that are correct for this output
            Y_temp = np.array([1 if k == j else 0 for k in Y])
            dw = (1 / m) * np.dot((A - Y_temp).T, X)
            db = (1/m) * np.sum(A-Y_temp)

            # update
            w[j] = w[j] - learning_rate * dw
            b = b - learning_rate * db

        if i % (num_interations / 10) == 0:
            print("loop:", i)

    pred = predict(w, b, X)
    conf_mat = confusion_matrix(Y, pred)
    print("Confusion Matrix =\n", conf_mat)
    print("train accuracy: {} %".format((np.trace(conf_mat) / np.sum(conf_mat)) * 100))

    if x_test is not None:
        pred = predict(w, b, x_test)
        conf_mat = confusion_matrix(y_test, pred)
        print("Confusion Matrix =\n", conf_mat)
        print("test accuracy: {} %".format((np.trace(conf_mat) / np.sum(conf_mat)) * 100))


def predict(w, b, X, activation_function='sigmoid'):

    m = len(X)

    pred = np.zeros(m)
    pred_val = np.zeros(m)

    for j, output in enumerate(w):

        # Make a prediction
        A = []
        if activation_function == 'sigmoid':
            A = sigmoid(np.dot(X, w[j]) + b)
        elif activation_function == 'relu':
            A = relu(np.dot(X, w[j]) + b)
        elif activation_function == 'tanh':
            A = tanh(np.dot(X, w[j]) + b)

        # Use the prediction with the highest confidence
        for i in range(A.shape[0]):
            if A[i] > pred[i]:
                pred[i] = A[i]
                pred_val[i] = j
    return pred_val


if __name__ == "__main__":
    print("Running Non-Binary Neural Net on Test Case")
    # Trivial Example for testing
    x = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 1]])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    # neural_network(x, y, [x.shape[0], 10, 1], activation_function='tanh', num_interations=50)

    print("\nRunning Non-Binary Neural Net on CIFAR-10")
    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset_full(5000)
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    neural_network(train_x, train_y, test_x, test_y, 'sigmoid', num_interations=1000)

    # Hypothesis idea, affect of bias weights and number of samples!
