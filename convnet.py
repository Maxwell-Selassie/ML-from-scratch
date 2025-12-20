import numpy as np

class ConvLayer:
    '''Convolutional layer: applies 128 filters of size 4*4 withe stride of 1 and no padding'''
    def __init__(self, num_filters=128, filter_size=4, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # initialize filters with random weights (He initialization)
        self.filters = np.random.randn(num_filters, filter_size,filter_size) * np.sqrt(2.0 / (filter_size * filter_size))
        self.bias = np.zeros(num_filters)

    def forward(self, X):
        '''Forward pass: convolve input with filters
        X shape: (batch_size, height, width, channels) - assuming a single channel input
        '''
        batch_size, height, width, channels = X.shape

        # add padding to input
        x_padded = np.pad(X,((0,0) (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        # calculate output dimensions
        out_height = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.filter_size) // self.stride + 1

        # initialize outputs
        output = np.zeros((batch_size, out_height, out_width, self.num_filters))

        # apply each filter to the output
        for f in range(self.num_filters):
            for i in range(out_height):
                for j in range(out_width):
                    # extract the patch from input
                    