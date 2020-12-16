import random
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from Utils.load_cifar10_keras import load_dataset, load_dataset_aruco
from Utils.reshape_data import reshape
import matplotlib.pyplot as plt
import pickle


class Neuron:
    def __init__(self, links, weights):
        self.active = 0
        self.weights = weights
        self.error = 0


class Layer:
    def __init__(self, size, links):

        self.active = np.array([0 for _ in range(size)], dtype=np.float64)
        self.error = np.array([0 for _ in range(size)], dtype=np.float64)
        self.bias = np.array([0 for _ in range(size)], dtype=np.float64)
        self.weights = np.array([[random.uniform(-.05, .05) for _ in range(links)] for _ in range(size)], dtype=np.float64)

        # self.neurons = np.array([Neuron(links, self.weights[i]) for i in range(size)])


class NeuralNetwork:
    def __init__(self, max_iters=2500):
        self.layers = []
        self.costs = []
        self.max_iters = max_iters
        self.performance_test = []
        self.performance_train = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fwd_pass(self, input):
        # Fast Way
        self.layers[0].active = self.sigmoid(np.dot(self.layers[0].weights, input) + self.layers[0].bias)

        # Note i is off by one
        for i, layer in enumerate(self.layers[1:]):
            layer.active = self.sigmoid(np.dot(layer.weights, self.layers[i].active) + layer.bias)

    def error_output(self, output):
        # Fast Way
        self.layers[-1].error = self.layers[-1].active * (1 - self.layers[-1].active) * (output - self.layers[-1].active)
        # self.layers[-1].bias = .5 * sum(self.layers[-1].active - output)

        for i, layer in reversed(list(enumerate(self.layers[:-1]))):
            z = np.sum(self.layers[i + 1].weights[:, :].T * self.layers[i + 1].error, axis=1)
            layer.error = layer.active * (1 - layer.active) * z
            # print(self.layers[i + 1].error.shape)
            # print(layer.active.shape)
            # layer.bias = .5 * sum(layer.active - self.layers[i + 1].error)

    def update_weights(self, input):
        alpha = 1
        # Fast Way
        self.layers[0].weights += alpha * np.outer(self.layers[0].error, input)

        for i, layer in enumerate(self.layers[1:]):
            layer.weights += alpha * np.outer(layer.error, self.layers[i].active)

        if should_print:
            for layer in self.layers:
                for weight, neuron in zip(layer.weights, layer.neurons):
                    print("fast", weight)
                    print("slow", neuron.weights)

    def predict(self, input):
        # fwd_pass(input, layers)
        # return layers[-1].neurons[0].active

        pred = np.zeros(len(input))

        A = []
        for i in input:
            self.fwd_pass(i)
            # A.append(layers[-1].neurons[0].active)
            A.append(self.layers[-1].active)
        # print(A)
        for i, p in enumerate(A):
            pred[i] = 1 if p > .25 else 0

        return pred

    def fit(self, X, Y, layers, x_test=None, y_test=None):
        self.layers = layers
        tic_o = time.perf_counter()

        # Number of iterations
        f_time, e_time, u_time = 0, 0, 0
        old_cost = 0
        for i in range(self.max_iters):
            total_cost = 0
            for example, answer in zip(X, Y):
                # tic = time.perf_counter()
                self.fwd_pass(example)
                # f_time += time.perf_counter() - tic

                # tic = time.perf_counter()
                self.error_output(answer)
                # e_time += time.perf_counter() - tic

                # tic = time.perf_counter()
                self.update_weights(example)
                # u_time += time.perf_counter() - tic

                total_cost += self.layers[-1].error
            self.costs.append(total_cost)

            # if x_test is not None:
            #     pred = self.predict(X)
            #     self.performance_train.append(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100)
            #     pred = self.predict(x_test)
            #     self.performance_test.append(100 - np.mean(np.abs(np.array(pred) - np.array(y_test))) * 100)
            if i % 100 == 0:
                pred = self.predict(X)
                print("Loop:", i, "Cost:", np.squeeze(total_cost), "Time Left:",
                      (self.max_iters - i - 1) * (time.perf_counter() - tic_o) / (i + 1),
                      "train accuracy: {} %".format(100 - np.mean(np.abs(np.array(pred) - np.array(Y))) * 100))

                # if abs(old_cost - total_cost) < .001:
                #     break
                # old_cost = total_cost

            # print(f_time, e_time, u_time)

        print("Took ", time.perf_counter() - tic_o, " seconds to run.")
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
    neural_network = NeuralNetwork()
    neural_network.fit(x, y, [Layer(20, len(x[0])), Layer(7, 20), Layer(5, 7), Layer(1, 5)])

    prediction = neural_network.predict(x)
    print("Confusion Matrix =\n", confusion_matrix(y, prediction))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(y))) * 100))

    # Run on cifar-10
    train_x, train_y, test_x, test_y = load_dataset(1000)
    temp_x = test_x
    temp_y = test_y
    train_x, train_y, test_x, test_y = reshape(train_x, train_y, test_x, test_y)
    neural_network = NeuralNetwork(500)
    # neural_network = pickle.load(open('trained_nn_full.p', "rb"))
    neural_network.fit(train_x, train_y, [Layer(20, len(train_x[0])), Layer(20, 20), Layer(1, 20)], test_x, test_y)

    prediction = neural_network.predict(train_x)
    print("Confusion Matrix =\n", confusion_matrix(train_y, prediction))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(train_y))) * 100))

    prediction = neural_network.predict(test_x)
    print("Confusion Matrix =\n", confusion_matrix(test_y, prediction))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(np.array(prediction) - np.array(test_y))) * 100))
    #
    pickle.dump(neural_network, open("trained_nn_full.p", "wb"))
    #
    # plt.plot(neural_network.costs)
    # plt.show()

    plt.figure(1)
    for i, (pred, truth) in enumerate(zip(prediction, test_y)):
        if pred == truth and pred == 1:
            plt.subplot(2, 2, 1)
            plt.imshow(temp_x[i])
            plt.title("True Positive")
        if pred == truth and pred == 0:
            plt.subplot(2, 2, 4)
            plt.imshow(temp_x[i])
            plt.title("True Negative")
        if pred != truth and pred == 1:
            plt.subplot(2, 2, 3)
            plt.imshow(temp_x[i])
            plt.title("False Positive")
        if pred != truth and pred == 0:
            plt.subplot(2, 2, 2)
            plt.imshow(temp_x[i])
            plt.title("False Negative")
    plt.figure(2)
    test = np.convolve(neural_network.performance_test, np.ones(10) / 10, mode='valid')
    train = np.convolve(neural_network.performance_train, np.ones(10) / 10, mode='valid')
    plt.plot(test)
    plt.plot(train)
    plt.legend(['Test Performance', 'Training Performance'])
    plt.title("Accuracy vs Number of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Percent Accuracy")
    plt.show()


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

    plt.plot(neural_network.costs)
    plt.show()
