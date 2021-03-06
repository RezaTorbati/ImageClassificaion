from keras.datasets import cifar10
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from pprint import pprint


# Loads the CIFAR-10 Dataset and gets it ready for binary classification
def load_dataset(num=None):
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_x, train_y = random_equalize(train_x, train_y, num)
    test_x, test_y = random_equalize(test_x, test_y)

    return train_x, train_y, test_x, test_y


def load_dataset_aruco(num=None):
    try:
        true_values = pd.read_csv("../Aruco/True.csv")
        true_values['class'] = 1
        false_values = pd.read_csv("../Aruco/False.csv")
        false_values['class'] = 0
    except:
        true_values = pd.read_csv("Aruco/True.csv")
        true_values['class'] = 1
        false_values = pd.read_csv("Aruco/False.csv")
        false_values['class'] = 0

    aruco_values = true_values.append(false_values)
    aruco_values = aruco_values.sample(frac=1).reset_index(drop=True)

    train = aruco_values.sample(frac=0.8)
    test = aruco_values.drop(train.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_y = train['class'].to_numpy()
    train_x = train.drop('class', axis=1).to_numpy()
    test_y = test['class'].to_numpy()
    test_x = test.drop('class', axis=1).to_numpy()

    if num:
        train_x = train_x[:num]
        train_y = train_y[:num]

    train_x = train_x / 255
    test_x = test_x / 255

    return train_x, train_y, test_x, test_y


# Loads the CIFAR-10 Dataset with all classes intact
def load_dataset_full(num=None):
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    # Sample the data such that there is an equal number of each class.
    train_x_new = []
    train_y_new = []
    for unique in np.unique(train_y):
        correct = np.where(np.array(train_y) == unique)[0]

        if num is None:
            first = [i for i in range(len(correct))]
        else:
            num_element = int(num / len(np.unique(train_y)))
            first = [i for i in range(num_element)]

        x_correct = np.take(np.take(train_x, correct, axis=0), first, axis=0)
        y_correct = np.take(np.take(train_y, correct, axis=0), first, axis=0)
        try:
            train_x_new = np.concatenate((train_x_new, x_correct))
            train_y_new = np.concatenate((train_y_new, y_correct))
        except ValueError:
            # First Elements
            train_x_new = x_correct
            train_y_new = y_correct

    train_x, train_y = shuffle(train_x_new, train_y_new)
    return train_x, train_y, test_x, test_y


# Converts x and y to binary output and equalizes number of true and false samples.
def random_equalize(x, y, num_samples=None):
    y = [1 if i == 0 else 0 for i in y]
    correct = np.where(np.array(y) == 1)[0]
    incorrect = np.where(np.array(y) != 1)[0]

    if num_samples is None:
        first = [i for i in range(len(correct))]
    else:
        first = [i for i in range(num_samples)]
    x_correct = np.take(np.take(x, correct, axis=0), first, axis=0)
    x_incorrect = np.take(np.take(x, incorrect, axis=0), first, axis=0)
    y_correct = np.take(np.take(y, correct, axis=0), first, axis=0)
    y_incorrect = np.take(np.take(y, incorrect, axis=0), first, axis=0)

    x_train = np.concatenate((x_correct, x_incorrect))
    y_train = np.concatenate((y_correct, y_incorrect))

    new_x, new_y = shuffle(x_train, y_train)
    return new_x, new_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_dataset_aruco()
    print("Aruco")
    print(train_y)
    print(train_x)
    print(test_y)
    print(test_x)

    train_x, train_y, test_x, test_y = load_dataset()
    print("Cifar")
    print(train_y)
    print(train_x)
    print(test_y)
    print(test_x)