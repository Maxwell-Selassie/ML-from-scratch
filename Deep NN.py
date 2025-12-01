import numpy as np
class DeepNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize weights for a deep network with arbitrary number of layers.
        
        layer_sizes: list like [input_size, hidden1, hidden2, ..., output_size]
        
        For each layer, we initialize W and b. The key difference from shallow networks:
        we can now approximate much more complex non-linear functions through composition.
        """
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.params = {}
        
        # Initialize parameters for each layer
        for i in range(1, self.num_layers):
            # He initialization helps with deeper networks
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * \
                        np.sqrt(2.0 / layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))
    
    def relu(self, x):
        """ReLU activation for hidden layers."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative for backpropagation."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax for output layer (multi-class classification)."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        For each hidden layer: Z = A_prev·W + b, then A = ReLU(Z)
        For output layer: Z = A_prev·W + b, then A = Softmax(Z)
        
        This composition of functions is why deep networks can learn complex patterns:
        f(x) = softmax(ReLU(...ReLU(ReLU(x·W1 + b1)·W2 + b2)·W3 + b3))
        """
        self.cache = {'A0': X}  # Store all intermediate activations for backprop
        A = X
        
        for i in range(1, self.num_layers - 1):
            # Hidden layers: linear transformation + ReLU
            Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
            A = self.relu(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
        
        # Output layer: linear transformation + Softmax
        i = self.num_layers - 1
        Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
        A = self.softmax(Z)
        self.cache[f'Z{i}'] = Z
        self.cache[f'A{i}'] = A
        
        return A
    
    def backward(self, Y, m):
        """
        Backpropagation through all layers using chain rule.
        
        Key insight: We propagate gradients backward, layer by layer.
        dL/dW_i = dL/dZ_i · dZ_i/dW_i = A_{i-1}^T · dZ_i
        dL/dA_{i-1} = dL/dZ_i · dZ_i/dA_{i-1} = dZ_i · W_i^T
        
        This allows us to compute gradients for all parameters efficiently.
        """
        gradients = {}
        
        # Start from output layer: dL/dZ
        i = self.num_layers - 1
        dZ = self.cache[f'A{i}'] - Y  # Softmax + cross-entropy derivative
        
        # Backpropagate through each layer
        for i in range(self.num_layers - 1, 0, -1):
            # Gradient w.r.t. weights and bias
            gradients[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / m
            gradients[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Gradient w.r.t. previous layer's activation (if not input layer)
            if i > 1:
                dA = np.dot(dZ, self.params[f'W{i}'].T)
                dZ = dA * self.relu_derivative(self.cache[f'Z{i-1}'])
        
        return gradients
    
    def update_weights(self, gradients):
        """Update all weights using computed gradients."""
        for i in range(1, self.num_layers):
            self.params[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.params[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']
    
    def train(self, X, Y, epochs=100):
        """Training loop for deep network."""
        m = X.shape[0]
        for epoch in range(epochs):
            output = self.forward(X)
            gradients = self.backward(Y, m)
            self.update_weights(gradients)
    
    def predict(self, X):
        """Inference: just forward pass."""
        return self.forward(X)
    

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    Y = np.eye(3)[np.random.randint(0, 3, 100)]  # 100 samples, 3 classes (one-hot encoded)
    
    print("=" * 60)
    print("DEEP NEURAL NETWORK")
    print("=" * 60)
    # Architecture: 10 input → 64 hidden → 32 hidden → 16 hidden → 3 output
    deep_nn = DeepNeuralNetwork(layer_sizes=[10, 64, 32, 16, 3])
    deep_nn.train(X, Y, epochs=100)
    predictions = deep_nn.predict(X[:5])
    print("Sample predictions shape:", predictions.shape)
    print("First prediction (probabilities):", predictions[0])