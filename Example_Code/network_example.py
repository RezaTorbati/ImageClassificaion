import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Utils.reshape_data import reshape_example
from Utils.load_cifar10_keras import load_dataset
import random

'''From: 
https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e'''

def initialize_with_zeros(dim):
    w = np.array([random.uniform(-.05, .05) for _ in range(dim)])
    print(w.shape)
    b = 0

    return w, b


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f' % (i, cost))

        params = {'w' : w, 'b': b}
        grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2500, learning_rate=0.5, print_cost=False):
    # Initialize parameters with 0s
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrive parameters w, b from dictionary
    w = parameters['w']
    b = parameters['b']

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print test/train errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {'costs': costs,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}

    return d


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_dataset(250)
    train_x, train_y, test_x, test_y = reshape_example(train_x, train_y, test_x, test_y, reshape_y=False)

    d = model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=.005, print_cost=True)

    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.show()
