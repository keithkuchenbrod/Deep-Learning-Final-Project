import numpy as np

class CrossEntropy(object):

    def __init__(self):
        #input is usually softmax output
        self.input, self.input_grad = None, None

    def hot_encode(self, x):
        """Used to create a 1 - hot encoded label list to calculate the derivative of the cross-entropy loss function
        :param x: label list
        :return: 1 - hot encoded label list
        """
        z = np.zeros(shape=(self.input.shape[0], self.input.shape[1], x.max()+1))
        for i in range(x.shape[0]):
            z[i, 0, x[i]] = 1.
        return z

    def forward(self, x, **kwargs):
        """Cross-Entropy:
        loss = -log(p of true label), p = probability
        since there is multiple images == batch_size going through network the avg loss of that batch is returned
        :param x: input from softmax
        :return: The cross-entropy loss
        """
        target = kwargs.get('target', None)
        self.input = x
        log = -np.log(x[range(x.shape[0]),:,target])
        loss = np.sum(log, axis=0) / x.shape[0]
        return loss[0]

    def backward(self, upstream_grad, **kwargs):
        """-(target/input)
        :param upstream_grad: gradient of output
        :return: gradient of input
        """
        target = kwargs.get('target', None)
        target = self.hot_encode(target)
        d_cross = (-target/self.input) + ((1-target)/(1-self.input))
        self.input_grad = d_cross * upstream_grad
        return self.input_grad
