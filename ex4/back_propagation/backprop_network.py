import random

import numpy as np

import math

from scipy import special



"""
GOAL: Implement the back-propagation algorithm for training a neural network.
      We will work with the MNIST data set that consists 0f 60000 28x28 gray scale
      images with values of 0 to 1 in each pixel (0 - white, 1 - black).
      The optimization problem we consider is of a neural network with ReLu activations and
      the cross entropy loss. 
"""


class Network(object):



    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)

                        for x, y in zip(sizes[:-1], sizes[1:])]



    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,

            test_data):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)

        training_accuracy=np.zeros(epochs)
        training_loss=np.zeros(epochs)
        test_accuracy=np.zeros(epochs)



        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [

                training_data[k:k+mini_batch_size]

                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))


        #code for question 2B:

            #test_accuracy[j]=self.one_label_accuracy(test_data)
            #training_accuracy[j]=self.one_hot_accuracy(training_data)
            #training_loss[j]=self.loss(training_data)

        #return test_accuracy,training_accuracy,training_loss







    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw

                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb

                       for b, nb in zip(self.biases, nabla_b)]



    def backprop(self, x, y):
        """The function receives as input a 784 dimensional

        vector x and a one-hot vector y.

        The function should return a tuple of two lists (db, dw)

        as described in the assignment pdf. """

        L=self.num_layers

        #forward pass
        v=[]
        z=[]
        z.append(x) #z0=x
        for i in range(L-1):
            v.append(np.dot(self.weights[i],z[-1])+self.biases[i])
            z.append(relu(v[i]))

        #backward pass
        delta=[]
        delta.append(self.loss_derivative_wr_output_activations(v[-1],y))#delta_L
        for i in range(L-3,-1,-1): #a little bit different calculation because of dimension issues
            delta.insert(0,np.dot(np.transpose(self.weights[i+1]),delta[0])*(relu_derivative(v[i])))


        #output
        dw=[]
        for i in range(L-1):
            dw.append(np.dot(delta[i],np.transpose(z[i]))) #z[i]=z_i-1 in our terms (indexing issues...)


        return delta, dw #db=delta



    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)

         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))



    def one_hot_accuracy(self,data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))

                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))





    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x)+b)

            layer += 1

        return x



    def loss(self, data):

        """Return the loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))



    def output_softmax(self, output_activations):

        """Return output after softmax given output before softmax"""

        #PAY ATTENTION: I've changed the implementation because of overflow during calculations
        #output_exp = np.exp(output_activations)
        #return output_exp/output_exp.sum()

        return special.softmax(output_activations)


    def loss_derivative_wr_output_activations(self, output_activations, y):


        return self.output_softmax(output_activations)-y





def relu(z):
    """
        Implements the ReLU activation function.

        Input:
        - z (float or numpy array): Input value or array.

        Returns:
        - float or numpy array: Result of the ReLU activation function.
        """
    # The ReLU activation function: max(0, z)
    # Multiplies z by the indicator function (z > 0)
    return (z*(z>0))



def relu_derivative(z):
    """
        Computes the derivative of the ReLU activation function.

        Parameters:
        - z (float or numpy array): Input value or array.

        Returns:
        - float or numpy array: Result of the derivative of the ReLU activation function.
        """
    # The derivative of the ReLU activation function: 1 if z > 0, 0 otherwise
    # Multiplies 1 by the indicator function (z > 0)
    return (1.*(z > 0))

