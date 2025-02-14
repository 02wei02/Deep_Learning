from .layer import *
import numpy as np



class Network(object):
    def __init__(self):
        self.fc1 = FullyConnected(28*28, 1024)  # First layer (input size, hidden layer size)
        self.act1 = Activation1()  # Activation function (ReLU)
        self.fc2 = FullyConnected(1024, 10)  # Second layer (hidden layer size, output size)
        # self.act2 = Activation1()
        # self.fc3 = FullyConnected(100, 10)
        # self.act3 = Sigmoid()
        # self.fc4 = FullyConnected(40, 10)
        self.loss = SoftmaxWithLoss()  # Softmax with cross-entropy loss

    def forward(self, input, target):
        h1 = self.fc1.forward(input)  # Forward pass through the first fully connected layer
        a1 = self.act1.forward(h1)    # Apply ReLU activation
        h2 = self.fc2.forward(a1)     # Forward pass through the second fully connected layer
        # a2 = self.act2.forward(h2)
        # h3 = self.fc3.forward(a2)
        # a3 = self.act3.forward(h3)
        # h4 = self.fc4.forward(a3)
        pred, loss = self.loss.forward(h2, target)  # Softmax and loss
        return pred, loss

    def backward(self):
        loss_grad = self.loss.backward()  # Gradient of loss
        # h4_grad = self.fc4.backward(loss_grad)  # Backpropagate through second layer
        # a3_grad = self.act3.backward(h4_grad)
        # h3_grad = self.fc3.backward(loss_grad)  # Backpropagate through second layer
        # a2_grad = self.act2.backward(h3_grad)
        h2_grad = self.fc2.backward(loss_grad)  # Backpropagate through second layer
        a1_grad = self.act1.backward(h2_grad)   # Backpropagate through ReLU
        h1_grad = self.fc1.backward(a1_grad)    # Backpropagate through first layer
        return h1_grad

    def update(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)
        # self.fc3.update(lr)
        # self.fc4.update(lr)