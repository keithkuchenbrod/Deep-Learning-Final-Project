import numpy as np
import time

class Network(object):

    def __init__(self, layers=None, optimizer=None, **kwargs):
        """
        Would like to eventually add different layers, optimizers, loss functions, and weight initializers
        :param layers: a list of layers that represent network model
        :param optimizer: optimizer used to update weights
        :param kwargs: optional -- weight_initializer=None since there is only one choice
        """

        self.layers = layers
        self.optimizer = optimizer
        self.weight_initializer = kwargs.get('weight_initializer', None)

    def forward(self, x, target, mode):
        """Calls forward function of all layers
        :param x: input
        :param target: input labels
        :return: loss(x)= last value calculated by the loss function and prediction values
        """
        for layer in self.layers:
            if mode is 'eval' and layer.__class__.__name__ is not 'Dropout':
                x = layer.forward(x, target=target)
            elif mode is 'train':
                x = layer.forward(x, target=target)
        predictions = np.argmax(self.layers[-1].input, axis=2)
        return x, predictions

    def backward(self, target):
        """Back pass calculation
        Passes upstream gradient to the backward pass calculation of the previous layer
        :param target: input labels
        """
        upstream_grad = np.ones(1)
        for layer in reversed(self.layers):
            if layer.__class__.__name__ is not 'Dropout':
                upstream_grad = layer.backward(upstream_grad, target=target)

    def apply(self):
        """Applies Xavier uniform initialization to all weights and biases by default"""
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                bound = 1./(layer.weight.shape[1]**(1/2))
                layer.weight = np.random.uniform(-bound, bound, layer.weight.shape)
                if layer.bias is not None:
                    layer.bias = np.random.uniform(-bound, bound, layer.bias.shape)

    def train(self, x_train, y_train, batch_size, epochs, **kwargs):
        """Trains neural network via back propagation
        :param x_train: training data
        :param y_train: training data labels
        :param batch_size: amount of data going through network at once
        :param epochs: number of times the entire data set runs through the network
        :param kwargs: optional params -- shuffle=True/False
        :return: dictionary containing loss and predictions both which are lists
        """

        #Initialize weights (Could possible change how this works)
        self.apply()

        #Values for history
        avg_loss, acc = [], []

        for e in range(epochs):
            error = 0
            t0 = time.time()
            predictions, loss = [], 0

            #Shuffle training data every epoch
            if kwargs.get('shuffle'):
                rand = np.random.permutation(batch_size)
                x_train, y_train = x_train[rand], y_train[rand]

            for batch in range(x_train.shape[0]):

                #Forward pass
                loss, pred = self.forward(x_train[batch], target=y_train[batch], mode='train')

                #Sum up loss and add predictions to array to be used for avg_loss and accuracy
                error += loss
                predictions.append(pred.T)

                #Back pass
                self.backward(target=y_train[batch])

                #Update weights
                self.optimizer.step(self.layers)

            avg_loss.append(error/x_train.shape[0])
            predictions = np.concatenate(predictions)
            result = np.concatenate(y_train - predictions)
            acc.append((1 - np.count_nonzero(result) / result.shape[0]))
            print("Epoch: {}\tAccuracy: {}\tLoss: {}\tTime elapsed: {} min".format(e+1, acc[-1], avg_loss[-1], (time.time()-t0)/60))

        history = {'loss': avg_loss, 'accuracy': acc}
        return history

    def evaluate(self, x_test, y_test, batch_size):
        """Evaluates trained model on new data not in training set
        :param x_test: test data
        :param y_test: test data labels (used to calculate accuracy and loss)
        :param batch_size: amount of data going through network at once
        :return: evaluation accuracy and loss (both are single values)
        """


        t0 = time.time()
        predictions = []
        error = 0

        for batch in range(x_test.shape[0]):

            #Forward pass
            loss, pred = self.forward(x_test[batch], target=y_test[batch], mode='eval')

            #Sum up loss and add predictions to array to be used for avg_loss and accuracy
            error += loss
            predictions.append(pred.T)

        avg_loss = error/x_test.shape[0]
        predictions = np.concatenate(predictions)
        result = np.concatenate(y_test - predictions)
        acc = 1 - np.count_nonzero(result) / result.shape[0]
        print("Evaluation: Accuracy: {}\tLoss: {}\tTime elapsed: {} min".format(acc, avg_loss, (time.time()-t0)/60))

        history = {'loss': avg_loss, 'accuracy': acc}
        return history






