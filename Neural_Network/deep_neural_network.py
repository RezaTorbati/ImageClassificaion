import random
import numpy as np
from sklearn.metrics import confusion_matrix
import time

class Neuron:
    def __init__(self, links):
        self.active = 0
        self.weights = np.array([random.uniform(-.05, .05) for _ in range(links)])
        self.error = 0


class Layer:
    def __init__(self, size, links):
        self.neurons = np.array([Neuron(links) for _ in range(size)])
        self.bias = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) - (1 - sigmoid(z))


def fwd_pass(input, layers):
    # Put the input into the input layer
    for neuron in layers[0].neurons:
        weighted_sum = np.dot(neuron.weights, input)
        neuron.active = sigmoid(weighted_sum)

    # Note i is off by one
    for i, layer in enumerate(layers[1:]):
        for neuron in layer.neurons:
            weighted_sum = np.dot(neuron.weights, [n.active for n in layers[i].neurons])
            neuron.active = sigmoid(weighted_sum)


def error_output(output, layers):
    # Put the input into the input layer
    for neuron in layers[-1].neurons:
        neuron.error = neuron.active * (1 - neuron.active) * (output - neuron.active)

    for i, layer in reversed(list(enumerate(layers[:-1]))):
        for j, neuron in enumerate(layer.neurons):
            forward_error = 0
            for k, past_neuron in enumerate(layers[i + 1].neurons):
                forward_error += past_neuron.error * past_neuron.weights[j]
            neuron.error = neuron.active * (1 - neuron.active) * forward_error


def update_weights(input, layers):
    alpha = .25

    # Put the input into the input layer
    for neuron in layers[0].neurons:
        for j, link in enumerate(input):
            neuron.weights[j] += alpha * neuron.error * link

    # Note i off by one
    for i, layer in enumerate(layers[1:]):
        for neuron in layer.neurons:
            links = [n.active for n in layers[i].neurons]
            for j, link in enumerate(links):
                neuron.weights[j] += alpha * neuron.error * link


def predict(input, layers):
    # fwd_pass(input, layers)
    # return layers[-1].neurons[0].active

    pred = np.zeros(len(input))

    A = []
    for i in input:
        fwd_pass(i, layers)
        A.append(layers[-1].neurons[0].active)

    for i, p in enumerate(A):
        pred[i] = 1 if p > .5 else 0

    return pred


def neural_network(X, Y, x_test=None, y_test=None):
    # Initalize weights to small number between -.05 and .05
    w = np.array([random.uniform(-.05, .05) for _ in range(len(X[0]))])
    b = 0

    layers = [Layer(10, len(X[0])), Layer(1, 10)]

    tic_o = time.perf_counter()
    # Number of iterations
    f_time, e_time, u_time = 0, 0, 0
    max_iters = 100
    for i in range(max_iters):
        for example, answer in zip(X, Y):

            tic = time.perf_counter()
            fwd_pass(example, layers)
            toc = time.perf_counter()
            f_time += toc - tic

            tic = time.perf_counter()
            error_output(answer, layers)
            toc = time.perf_counter()
            e_time += toc - tic

            tic = time.perf_counter()
            update_weights(example, layers)
            toc = time.perf_counter()
            u_time += toc - tic

        print(i, (max_iters - i - 1) * (time.perf_counter() - tic_o) / (i + 1))

    pred = predict(X, layers)
    print("Confusion Matrix =\n", confusion_matrix(Y, pred))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100))

    toc_o = time.perf_counter()
    print("Took ", toc_o - tic_o, " seconds to run.")
    print(f_time, e_time, u_time)


from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco
from Utils.reshape_data import reshape

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

    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset(100)
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    neural_network(train_x, train_y, test_x, test_y)