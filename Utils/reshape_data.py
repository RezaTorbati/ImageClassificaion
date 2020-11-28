import numpy as np


# Converts data from (image number, data) to (data, image_number)
# Example code does it in this order for some reason.
def reshape_example(train_x, train_y, test_x, test_y, reshape_y=True):
    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    train_x = train_x / 255
    test_x = test_x / 255

    if reshape_y:
        train_y = np.reshape(train_y, (len(train_y), 1))
        test_y = np.reshape(test_y, (len(test_y), 1))

    return train_x, train_y, test_x, test_y


# converts RGB images to flat arrays of pixel intensities
def refit(images):
    new_images = np.empty([images.shape[0], 3072])
    for i, image in enumerate(images):
        new_images[i] = np.ravel(image)
    return new_images


# Normalizes and formats the train and test data
def reshape(train_x, train_y, test_x, test_y):
    x_train = refit(train_x)
    y_train = np.ravel(train_y)
    x_train = x_train / 255

    x_test = refit(test_x)
    y_test = np.ravel(test_y)
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test
