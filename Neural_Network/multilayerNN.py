import pandas as pd
import numpy as np
import random
import copy

def getSigmoid(z):
    return 1 / (1 + np.exp(-z))

def getSigmoidPrime(z):
    return getSigmoid(z) * (1 - getSigmoid(z))


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.uniform(-.05, .05, (y, 1))  for y in sizes[1:]]
        self.weights = [np.random.uniform(-.05, .05, (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

    def train(self, trainData, epochs, batchSize, learningRate, testData = None):
        for i in range(0, epochs):
            random.shuffle(trainData)
            batches = (trainData[j:j+batchSize] for j in range(0, len(trainData), batchSize))

            for batch in batches:
                self.update(batch, learningRate)

            #add update on test            
            print("Epoch complete")

    def update(self, batch, learningRate):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            deltaNablaB, deltaNablaW = self.backprop(x,y)

            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]

        self.weights = [w - (learningRate / len(batch)) * nw for (w, nw) in zip(self.weights, nablaW)]
        self.biases = [b - (learningRate / len(batch)) * nb for (b, nb) in zip(self.biases, nablaB)]

    def backprop(self, x, y):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        outs = []

        #Feed Forward
        for b, w in zip(self.biases, self.weights):
            o = w @ activation + b
            outs.append(o)
            activation = getSigmoid(o)
            activations.append(activation)

        #Backward Pass
        delta = (activations[-1] - y) * getSigmoidPrime(outs[-1])
        nablaB[-1] = delta
        nablaW[-1] = delta @ activations[-2].transpose()

        for layer in range(2, self.layers):
            o = outs[-layer]
            sig = getSigmoidPrime(o)

            delta = (self.weights[-layer + 1].transpose() @ delta) * sig
            nablaB[-layer] = delta
            #print(activations[-layer - 1].transpose())
            #print(delta)

            #print(len(delta[0]))
            #print(len(activations[-layer - 1].transpose()))

            nablaW[-layer] = delta @ activations[-layer - 1].transpose()

        return (nablaB, nablaW)

    def getResult(self, data):
        for b, w in zip(self.biases, self.weights):
            data = getSigmoid(w @ data + b)
        return data

    def evaluate(self, testData):
        results = [(np.argmax(self.getResult(x)), y) for (x, y) in testData]
        return sum(int(x==y) for (x,y) in results)

# training examples
'''
df = pd.read_csv("../cifar-10-batches-py/data_batch_full.csv")
trueDf = df[df.label == 1] #gets all of the examples of an automobile
falseDf = df[df.label != 1].sample(5000) #gets 5000 examples without an automobile
falseDf['label'] = 0

trainDf = pd.concat([trueDf, falseDf])
yTrain = trainDf['label']
xTrain = trainDf
del xTrain['label']

xTrainAr = xTrain.to_numpy()
yTrainAr = yTrain.to_numpy()

#Test Examples
df2 = pd.read_csv("../cifar-10-batches-py/test_batch.csv")
testTrueDf = df2[df2.label == 1]
testFalseDf = df2[df2.label != 1].sample(1000)
testFalseDf['label'] = 0

testDf = pd.concat([testTrueDf, testFalseDf])
yTest = testDf['label']
xTest = testDf
del xTest['label']

xTestAr = xTest.to_numpy()
yTestAr = yTest.to_numpy()

##nn = NeuralNetwork((1024, 5, 6, 2))
##nn.train(list(zip(xTrainAr, yTrainAr)), 100, 16, .1)
##print(nn.evaluate(zip(xTestAr, yTestAr)))
'''

