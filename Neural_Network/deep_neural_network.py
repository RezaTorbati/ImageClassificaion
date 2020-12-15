import random
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco
from Utils.reshape_data import reshape


class Neuron:
    def __init__(self, links, weights):
        self.active = 0
        self.weights = weights
        self.error = 0


class Layer:
    def __init__(self, size, links):

        self.active = np.array([0 for _ in range(size)], dtype=np.float64)
        self.error = np.array([0 for _ in range(size)], dtype=np.float64)
        # self.bias = np.array([0 for _ in range(size)], dtype=np.float64)
        self.weights = np.array([[random.uniform(-.05, .05) for _ in range(links)] for _ in range(size)], dtype=np.float64)

        self.neurons = np.array([Neuron(links, self.weights[i]) for i in range(size)])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fwd_pass(input, layers):
    # # Easy to understand Way
    # # Put the input into the input layer
    # for neuron in layers[0].neurons:
    #     weighted_sum = np.dot(neuron.weights, input)
    #     neuron.active = sigmoid(weighted_sum)
    #
    # # # Note i is off by one
    # for i, layer in enumerate(layers[1:]):
    #     for neuron in layer.neurons:
    #         weighted_sum = np.dot(neuron.weights, [n.active for n in layers[i].neurons])
    #         neuron.active = sigmoid(weighted_sum)

    # Fast Way
    layers[0].active = sigmoid(np.dot(layers[0].weights, input))

    # Note i is off by one
    for i, layer in enumerate(layers[1:]):
        layer.active = sigmoid(np.dot(layer.weights, layers[i].active))


def error_output(output, layers):
    # # Easy to understand Way
    # # Put the input into the input layer
    # for neuron in layers[-1].neurons:
    #     neuron.error = neuron.active * (1 - neuron.active) * (output - neuron.active)
    #     # print(neuron.error)
    #
    # for i, layer in reversed(list(enumerate(layers[:-1]))):
    #     for j, neuron in enumerate(layer.neurons):
    #         forward_error = 0
    #         for k, past_neuron in enumerate(layers[i + 1].neurons):
    #             forward_error += past_neuron.error * past_neuron.weights[j]
    #         neuron.error = neuron.active * (1 - neuron.active) * forward_error
    #     # print("slow error", [neuron.error for neuron in layer.neurons])


    # Fast Way
    layers[-1].error = layers[-1].active * (1 - layers[-1].active) * (output - layers[-1].active)
    # print(layers[-1].error)
    # layers[-1].bias = .5 * sum(layers[-1].active - output)

    for i, layer in reversed(list(enumerate(layers[:-1]))):
        for j, neuron_error in enumerate(layer.error):
            forward_error = 0
            for k, past_neuron_error in enumerate(layers[i + 1].error):
                forward_error += past_neuron_error * layers[i + 1].weights[k, j]
            e = layer.active[j] * (1 - layer.active[j]) * forward_error
            layer.error[j] = e
            # layer.bias = .5 * sum(layer.active - layers[i + 1].error)
        # print('fast error', layer.error)


def update_weights(input, layers):
    alpha = .5

    # # Easy to understand Way
    # # Put the input into the input layer
    # for neuron in layers[0].neurons:
    #     for j, link in enumerate(input):
    #         neuron.weights[j] += alpha * neuron.error * link
    #
    #
    # # Note i off by one
    # for i, layer in enumerate(layers[1:]):
    #     for neuron in layer.neurons:
    #         links = [n.active for n in layers[i].neurons]
    #         for j, link in enumerate(links):
    #             neuron.weights[j] += alpha * neuron.error * link
    #     # print("slow weights", [neuron.weights for neuron in layer.neurons])

    # Fast Way
    layers[0].weights += alpha * np.outer(layers[0].error, input)

    for i, layer in enumerate(layers[1:]):
        layer.weights += alpha * np.outer(layer.error, layers[i].active)
        # print("fast weights", layer.weights)

    if should_print:
        for layer in layers:
            for weight, neuron in zip(layer.weights, layer.neurons):
                print("fast", weight)
                print("slow", neuron.weights)

def predict(input, layers):
    # fwd_pass(input, layers)
    # return layers[-1].neurons[0].active

    pred = np.zeros(len(input))

    A = []
    for i in input:
        fwd_pass(i, layers)
        # A.append(layers[-1].neurons[0].active)
        A.append(layers[-1].active)
    print(A)
    for i, p in enumerate(A):
        pred[i] = 1 if p > .25 else 0

    return pred


def neural_network(X, Y, x_test=None, y_test=None):
    layers = [Layer(20, len(X[0])), Layer(7, 20), Layer(5, 7), Layer(1, 5)]

    tic_o = time.perf_counter()
    # Number of iterations
    f_time, e_time, u_time = 0, 0, 0
    max_iters = 2000
    for i in range(max_iters):
        for example, answer in zip(X, Y):

            tic = time.perf_counter()
            fwd_pass(example, layers)
            f_time += time.perf_counter() - tic

            tic = time.perf_counter()
            error_output(answer, layers)
            e_time += time.perf_counter() - tic

            tic = time.perf_counter()
            update_weights(example, layers)
            u_time += time.perf_counter() - tic
        if i % 100 == 0:
            print(i, (max_iters - i - 1) * (time.perf_counter() - tic_o) / (i + 1))

        # print(f_time, e_time, u_time)

    pred = predict(X, layers)
    print("Confusion Matrix =\n", confusion_matrix(Y, pred))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100))

    if x_test is not None:
        pred = predict(x_test, layers)
        print("Confusion Matrix =\n", confusion_matrix(y_test, pred))
        print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100))

    toc_o = time.perf_counter()
    print("Took ", toc_o - tic_o, " seconds to run.")
    print(f_time, e_time, u_time)


if __name__ == "__main__":
    should_print = False

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
    # train_x, train_y, test_x, test_y = load_dataset(250)
    # train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    # neural_network(train_x, train_y, test_x, test_y)