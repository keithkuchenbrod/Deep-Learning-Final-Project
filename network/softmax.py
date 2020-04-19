import numpy as np

class SoftMax(object):

    def __init__(self):
        self.input, self.input_grad = None, None
        self.output = None

    def forward(self, x, **kwargs):
        """This calculates the stable softmax probabilities
        - shift all values by the max value in list --> x - max(x), x is a list of values
        - take the exp() of all values in the shifted list x --> exp(x)
        - get the probabilities by normalizing the values in x by dividing by the sum --> sum(x)
        :param x: The values from the previous layer
        :return: the softmax (stable) probabilities for the 10 classes of MNIST
        """
        self.input = x
        self.output = np.asarray([np.exp(x[i] - np.max(x[i]))/np.sum(np.exp(x[i] - np.max(x[i]))) for i in range(x.shape[0])])
        return self.output

    def backward(self, upstream_grad, **kwargs):
        """Derivative of softmax function
        The vectorized form of the softmax function is a jacobian matrix

        let p be softmax output:
        d_softmax = p[i](1-p[j]) for i == j
        d_softmax = -p[i]p[j] for i != j

        the full vectorized form in below:

        d_softmax = diag_s - dot(p, p.T)
        diag_s = a square jacobian matrix representing when i == j
        dot(p, p.T) = a square jacobian matrix representing when i != j

        :param upstream_grad: the gradient of the output
        :param kwargs: not use in this
        :return: the input gradient
        """
        diag_s = np.asarray([np.diagflat(self.output[i]) for i in range(self.output.shape[0])])
        p_t = np.reshape(self.output, newshape=(self.output.shape[0], self.output.shape[2], self.output.shape[1]))
        d_softmax = diag_s - self.output*p_t
        self.input_grad = np.matmul(upstream_grad, d_softmax)
        return self.input_grad
