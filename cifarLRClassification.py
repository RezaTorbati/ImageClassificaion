import pandas as pd
import numpy as np
import math
import scipy.stats
import copy
from sklearn.metrics import roc_auc_score


def cifar_lr():
    #Training Examples
    #commented out code below is for if you don't have the data_batch_full.csv because github won't allow files of that size
    df1 = pd.read_csv("cifar-10-batches-py/data_batch_1.csv")
    df2 = pd.read_csv("cifar-10-batches-py/data_batch_2.csv")
    df3 = pd.read_csv("cifar-10-batches-py/data_batch_3.csv")
    df4 = pd.read_csv("cifar-10-batches-py/data_batch_4.csv")
    df5 = pd.read_csv("cifar-10-batches-py/data_batch_5.csv")
    df = pd.concat([df1, df2, df3, df4, df5])

    # df = pd.read_csv("cifar-10-batches-py/data_batch_full.csv")
    df.insert(0,'bias',np.ones(len(df)))
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
    df2 = pd.read_csv("cifar-10-batches-py/test_batch.csv")
    df2.insert(0,'bias', np.ones(len(df2)))
    testTrueDf = df2[df2.label == 1]
    testFalseDf = df2[df2.label != 1].sample(1000)
    testFalseDf['label'] = 0

    testDf = pd.concat([testTrueDf, testFalseDf])
    yTest = testDf['label']
    xTest = testDf
    del xTest['label']

    xTestAr = xTest.to_numpy()
    yTestAr = yTest.to_numpy()


    #Linear Regression:
    weights = np.linalg.inv(xTrainAr.transpose() @ xTrainAr) @ xTrainAr.transpose() @ yTrainAr

    print("Linear Regression Metrics:")
    # Tests on training data
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    rss = 0
    tss = 0
    for i in range(0, len(yTrainAr)):
        tss += (yTrainAr[i] - yTrainAr.mean()) ** 2

        pred = xTrainAr[i] @ weights
        rss += (yTrainAr[i] - pred) ** 2

        sqnPred = 0
        if(pred > .5):
            sqnPred = 1

        if(yTrainAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTrainAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTrainAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    rsq = 1 - rss / tss
    print("R^2: ", rsq)
    k = len(weights - 1)
    n = len(yTrainAr)
    f = (rsq / k) / ((1 - rsq) / (n - k - 1))
    p = 1-scipy.stats.f.cdf(f, k, n - k - 1)
    print("F test P value: ", p)
    print("Train Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Train True positive rate: ", tpr)
    print("Train False positive rate: ", fpr)
    print()

    # Tests on test data
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(0, len(yTestAr)):
        pred = xTestAr[i] @ weights

        sqnPred = 0
        if(pred > .5):
            sqnPred = 1

        if(yTestAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTestAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTestAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    print("Test Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Test True positive rate: ", tpr)
    print("Test False positive rate: ", fpr)
    print()


    # Logistic Regression with Stochastic Gradient Descent:
    def getSigmoid(val):
        try:
            return 1 / (1 + math.exp(-val))
        except OverflowError:
            if val < 0:
                return 0
            else:
                return 1

    def getStochasticGrad(weights, x, y):
        grad = np.zeros(len(x))
        sig = getSigmoid(x @ weights)
        coef = sig - y
        for i in range(0, len(x)):
            grad[i] = (coef * x[i])
        return grad

    learningRate = .00000002
    logWeights = np.zeros(len(xTrainAr[0]))
    oldLogWeights = np.ones(len(xTrainAr[0]))

    samples = np.random.choice(len(yTrain), size = len(yTrain),replace = False)
    for i in samples:
        oldLogWeights = copy.deepcopy(logWeights)
        logWeights = logWeights - learningRate * getStochasticGrad(logWeights, xTrainAr[i], yTrainAr[i])
        #if np.linalg.norm(logWeights - oldLogWeights) < .000001:
        #    print("breaking ", count)
        #    break

    #print(np.linalg.norm(logWeights - oldLogWeights))

    print("\nLogistic Metrics with Stochastic Gradient Descent:")
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    pred = np.zeros(len(yTrainAr))
    for i in range(0, len(yTrainAr)):
        pred[i] = getSigmoid(xTrainAr[i] @ logWeights)

        sqnPred = 0
        if(pred[i] > .5):
            sqnPred = 1

        if(yTrainAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTrainAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTrainAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    print("AUC: ", roc_auc_score(yTrainAr, pred))
    print(tp, ", ", fn, ", ", tn, ", ", fp)
    print("Train Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Train True positive rate: ", tpr)
    print("Train False positive rate: ", fpr)
    print()

    tp = 0
    fn = 0
    tn = 0
    fp = 0
    pred = np.zeros(len(yTestAr))
    for i in range(0, len(yTestAr)):
        pred[i] = getSigmoid(xTestAr[i] @ logWeights)

        sqnPred = 0
        if(pred[i] > .5):
            sqnPred = 1

        if(yTestAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTestAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTestAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    print("AUC: ", roc_auc_score(yTestAr, pred))
    print(tp, ", ", fn, ", ", tn, ", ", fp)
    print("Test Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Test True positive rate: ", tpr)
    print("Test False positive rate: ", fpr)
    print()


    # Logistic Regression with Gradient Descent
    def getGrad(weights, xs, ys):
        gradMat = np.zeros((len(ys), len(xs[0])))
        for i in range(0, len(ys)):
            #sig = getSigmoid(xs[i] @ weights) - ys[i]
            #coef = sig - ys[i]
            gradMat[i] = (getSigmoid(xs[i] @ weights) - ys[i]) * xs[i]

        grad = np.sum(gradMat, axis = 0) / len(gradMat)
        return grad

    learningRate = .0000005
    logWeights = np.zeros(len(xTrainAr[0]))
    oldLogWeights = np.ones(len(xTrainAr[0]))

    count = 1
    print("Logistic Training...")
    while True:
        oldLogWeights = copy.deepcopy(logWeights)
        logWeights = logWeights - learningRate * getGrad(logWeights, xTrainAr, yTrainAr)
        if np.linalg.norm(logWeights - oldLogWeights) < .000001 or count > 100: #This is just going to run 100 times...
            break
        count += 1
        if count % 10 == 0:
            print(count ,"% finished")

    print("\nLogistic Metrics:")
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    pred = np.zeros(len(yTrainAr))
    for i in range(0, len(yTrainAr)):
        pred[i] = getSigmoid(xTrainAr[i] @ logWeights)

        sqnPred = 0
        if(pred[i] > .5):
            sqnPred = 1

        if(yTrainAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTrainAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTrainAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    print("AUC: ", roc_auc_score(yTrainAr, pred))
    print(tp, ", ", fn, ", ", tn, ", ", fp)
    print("Train Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Train True positive rate: ", tpr)
    print("Train False positive rate: ", fpr)
    print()

    tp = 0
    fn = 0
    tn = 0
    fp = 0
    pred = np.zeros(len(yTestAr))
    for i in range(0, len(yTestAr)):
        pred[i] = getSigmoid(xTestAr[i] @ logWeights)

        sqnPred = 0
        if(pred[i] > .5):
            sqnPred = 1

        if(yTestAr[i] == 1 and sqnPred == 1):
            tp += 1
        elif(yTestAr[i] == 1 and sqnPred == 0):
            fn += 1
        elif(yTestAr[i] == 0 and sqnPred == 0):
            tn += 1
        else:
            fp += 1

    print("AUC: ", roc_auc_score(yTestAr, pred))
    print(tp, ", ", fn, ", ", tn, ", ", fp)
    print("Test Accuracy: ", ((tp + tn) / (tp + fn + tn + fp)))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    print("Test True positive rate: ", tpr)
    print("Test False positive rate: ", fpr)
    print()


if __name__ == "__main__":
    cifar_lr()