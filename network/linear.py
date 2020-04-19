import numpy as np

class Linear(object):
    """This serves as the fully connected layer in a network"""

    def __init__(self, in_kernel, out_kernel, **kwargs):
        """
        :param in_kernel: the input size
        :param out_kernel: the output size
        :param kwargs: used for option parameter -- bias=True or bias=False
        """
        #Holds forward pass values
        self.input, self.input_grad = None, None

        #Intializes weights/bias with zeros
        self.shape = (in_kernel, out_kernel)
        self.weight, self.weight_grad = np.zeros((in_kernel, out_kernel)), np.zeros((in_kernel, out_kernel))

        #Condition to use bias or to not use
        self.bias = None
        if kwargs.get('bias', False):
            self.bias = np.zeros((1,out_kernel))

    def forward(self, x, **kwargs):
        """Forward pass calculation
        :param x: input
        :param kwargs: not used in this
        :return: output of forward pass calculation
        """
        if self.bias is not None:
            self.input = x
            return np.matmul(x, self.weight) + self.bias
        self.input = x
        return np.matmul(x, self.weight)

    def backward(self, upstream_grad, **kwargs):
        """Back pass calculation
        :param upstream_grad: the gradient of this layers output
        :param kwargs: not used in this
        :return: gradient of input
        """
        self.weight_grad = np.matmul(np.reshape(self.input,newshape=(self.input.shape[0],self.input.shape[2],self.input.shape[1])), upstream_grad)
        self.input_grad = np.matmul(upstream_grad, self.weight.T)
        return self.input_grad



