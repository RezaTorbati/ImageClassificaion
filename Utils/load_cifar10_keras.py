from keras.datasets import cifar10
from sklearn.utils import shuffle
import numpy as np
from pprint import pprint


# Loads the CIFAR-10 Dataset and gets it ready for binary classification
def load_dataset(num=None):
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_x, train_y = random_equalize(train_x, train_y, num)
    test_x, test_y = random_equalize(test_x, test_y, num)

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
