import numpy as np
class ShallowNeuralNetwork:
    def __init__(self, hidden_units, input_size, output_size, learning_rate=0.01):
        '''
        Initialize weights and biases for a shallow network
        '''
        # He initialization. 
        # This prevents gradient vanishing/exploding in neural networks
        self.w1 = np.random.randn(input_size, hidden_units) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_units))

        self.w2 = np.random.randn(hidden_units, output_size) * np.sqrt(2.0 / hidden_units)
        self.b2 = np.random.randn(1, output_size)

        self.learning_rate = learning_rate

    def relu(self, x):
        '''
        ReLU (Rectified Linear Unit) activation : max(0, x)
        '''
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        '''
        Derivative of ReLU : 1 if x > 0 else 0
        '''
        return (x > 0).astype(float)
    


    def softmax(self, x):
        '''
        softmax activation : exp(x) / sum(exp(x))
        '''
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, x):
        '''forward pass: compute predictions by passing data through network'''
        self.x = x
        self.z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward_prob(self, Y, m):
        '''Backpropagation: Compute gradients by chain rule, flowing backward through the network'''
        dz2 = self.a2 -  Y
        dw2 = np.dot(self.a1.T,dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.w2.T)

        dz1 = da1 * self.relu_derivative(self.z1)

        dw1 = np.dot(self.x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dw1, db1, dw2, db2

    