from __future__ import print_function
import mnist_loader
import network
from network import Network

# hyperparameters
learning_rate = 3.0
epochs = 30
mini_batch_size = 10

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784,30,10])
net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)
