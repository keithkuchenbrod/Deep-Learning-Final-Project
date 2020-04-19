import numpy as np

class ReLU(object):

    def __init__(self):
        self.input, self.input_grad = None, None

    def forward(self, x, **kwargs):
        """ReLU:
        x, x>0
        0, x<=0
        :param x: input
        :param kwargs: not used here
        :return: relu output
        """
        self.input = np.maximum(0, x)
        return self.input

    def backward(self, upstream_grad, **kwargs):
        """
        The derivative of ReLU:

        0 for x <= 0
        1 for x > 0

        :param upstream_grad: layer output
        :return: the derivative of ReLU
        """
        d_ReLU = 1. * ( self.input > 0 )
        self.input_grad = d_ReLU * upstream_grad
        return self.input_grad

