import numpy as np

class _Layer(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError

class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)  # Xavier initialization
        self.bias = np.zeros((1, out_features))
        self.input = None

    def forward(self, input):
        self.input = input / 255.0
        return np.dot(self.input, self.weight) + self.bias  # Linear transformation

    def backward(self, output_grad):
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        return np.dot(output_grad, self.weight.T)

    def update(self, lr):
        # Update weights using gradient descent
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad

class Activation1(_Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)  # ReLU activation

    def backward(self, output_grad):
        input_grad = np.zeros_like(self.input)
        input_grad[self.input > 0] = 1  # Derivative of ReLU: 1 for input > 0, else 0
        return output_grad * input_grad  # Element-wise multiply by output_grad

class SoftmaxWithLoss(_Layer):
    def __init__(self):
        self.y_pred = None
    def forward(self, input, target):
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.target = target
        epsilon = 1e-15 
        self.y_pred = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        log_prod = np.log(self.y_pred + epsilon) * (target)
        your_loss = -np.sum(log_prod) / target.shape[0]
        return self.y_pred, your_loss

    def backward(self):
      grad_input =  (self.y_pred) - self.target
      return grad_input

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input  # Store the input for the backward pass
        
        # Clip input values to avoid overflow
        clipped_input = np.clip(input, -709, 709)  # np.exp cannot handle values < -709
        
        self.output = 1 / (1 + np.exp(-clipped_input))  # Compute the output
        return self.output

    def backward(self, dout):
        # Use the stored output to compute the gradient
        return dout * self.output * (1 - self.output)  # Calculate the gradient
