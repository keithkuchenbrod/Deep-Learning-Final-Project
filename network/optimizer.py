import numpy as np

class SGD(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):
        """Updates weights and bias vectors
        :param layers: all layers from network
        """
        for i in range(len(layers)):
            if layers[i].__class__.__name__ == 'Linear':
                #Update weights
                layers[i].weight -= self.learning_rate * (np.sum(layers[i].weight_grad, axis=0) / layers[i].weight.shape[0])

                #Update biases if they are used
                if layers[i].bias is not None and i<=len(layers)-1:
                    layers[i].bias -= self.learning_rate * (np.sum(layers[i+1].input_grad, axis=0)/layers[i+1].input_grad.shape[0])




