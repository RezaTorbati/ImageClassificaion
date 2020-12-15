from textBook import Network
import mnistLoader

training_data, validation_data, test_data = mnistLoader.load_data_wrapper()

print(type(training_data))
print(len(training_data))
print(len(training_data[0]))
print(len(training_data[1]))
print(training_data[:10])

net = Network([784, 30, 10])

net.SGD(training_data, 5, 5, 3.0, test_data=test_data)