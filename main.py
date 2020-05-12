from download_mnist import load
import network as net
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    """ Type B project number 4)
    Manually implement a fully-connected NN using Numpy (no automatic gradient). Train
    and test on MNIST dataset. The 90% accuracy should be achieved. 
    """
    #Load MNIST data

    x_train, y_train, x_test, y_test = load()
    x_train = x_train.reshape(60000, 28, 28)
    x_test = x_test.reshape(10000, 28, 28)
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    #Hyper params
    batch_size = 128
    epochs = 15
    learning_rate = 0.1

    #Shaping training data and labels into batches and normalizing data by dividing by 255
    x = np.asarray([np.reshape(x_train[batch_size*i:batch_size*(i+1)], newshape=(batch_size, 1, 784)) / 255 for i in range(x_train.shape[0]//batch_size)])
    y = np.asarray([y_train[batch_size*i:batch_size*(i+1)] for i in range(y_train.shape[0]//batch_size)])

    #Shaping testing data and labels into batches and normalizing data by dividing by 255
    x_test = np.asarray([np.reshape(x_test[batch_size*i:batch_size*(i+1)], newshape=(batch_size, 1, 784)) / 255 for i in range(x_test.shape[0]//batch_size)])
    y_test = np.asarray([y_test[batch_size*i:batch_size*(i+1)] for i in range(y_test.shape[0]//batch_size)])

    #Setting up model structure
    layers = [
        net.Linear(in_kernel=784, out_kernel=200, bias=True),
        net.Linear(200, 100, bias=True),
        net.ReLU(),
        net.Linear(100, 10, bias=True),
        net.ReLU(),
        net.Linear(10, 10, bias=True),
        net.SoftMax(),
        net.CrossEntropy()
    ]

    #Init new SGD optimizer
    optimizer = net.SGD(learning_rate=learning_rate)

    #Intit new network
    network = net.Network(layers=layers, optimizer=optimizer)

    #Training Network
    t0 = time.time()
    print("Starting training process...")
    history = network.train(x, y, batch_size, epochs, shuffle=True)

    #Testing Network
    print("Starting evaluation process...")
    evaluation = network.evaluate(x_test, y_test, batch_size)
    print("Total time elapsed: {} minutes".format((time.time() - t0)/60))

    #Plotting Training Loss
    ax1 = plt.subplot(311)
    ax1.plot(range(epochs), history['loss'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')

    #Plotting Training Accuracy
    ax2 = plt.subplot(313)
    ax2.plot(range(epochs), history['accuracy'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')

    #Displaying plot
    plt.show()





