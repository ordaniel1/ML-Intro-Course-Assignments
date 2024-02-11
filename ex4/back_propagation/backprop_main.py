import backprop_data
import backprop_network
import numpy as np
import matplotlib.pyplot as plt




# Load training and test data
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

# Initialize a neural network with architecture [784, 40, 10]
net = backprop_network.Network([784, 40, 10])

# Train the neural network using SGD with specified parameters
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


#Code for question 2B
def B(): #uncomment code in SGD function in backprop_network.py before calling to B()

    """
    Perform experiments for different learning rates and visualize the results.

    Uncomment code in the SGD function in backprop_network.py before calling B().
    """

    learning_rates=[0.001, 0.01, 0.1, 1, 10, 100]

    test_accuracy=[0 for i in range(len(learning_rates))]
    training_accuracy=[0 for i in range(len(learning_rates))]
    training_loss=[0 for i in range(len(learning_rates))]

    for i in range(len(learning_rates)):

        # Initialize a neural network with architecture [784, 40, 10]
        net = backprop_network.Network([784, 40, 10])

        # Run SGD with different learning rates and collect accuracy and loss data
        test_accuracy[i], training_accuracy[i], training_loss[i]=net.SGD(training_data, epochs=30, mini_batch_size=10,
                                                                    learning_rate=learning_rates[i], test_data=test_data)

    # Visualize test accuracy
    n=np.arange(0,30)
    fig, ax = plt.subplots()
    rate0001 = ax.plot(n, test_accuracy[0], "-r", label="0.001 learning rate")
    rate001 = ax.plot(n, test_accuracy[1], "-g", label="0.01 learning rate")
    rate01 = ax.plot(n, test_accuracy[2], "-b", label="0.1 learning rate")
    rate1 = ax.plot(n, test_accuracy[3], "-c", label="1 learning rate")
    rate10 = ax.plot(n, test_accuracy[4], "-m", label="10 learning rate")
    rate100= ax.plot(n, test_accuracy[5], "-k", label="100 learning rate")
    ax.legend(["learning rate=0.001", "learning rate=0.01", "learning rate=0.1",
          "learning rate=1" ,"learning rate=10","learning rate=100"])
    plt.show()


    # Visualize training accuracy
    fig, ax = plt.subplots()
    rate0001 = ax.plot(n, training_accuracy[0], "-r", label="0.001 learning rate")
    rate001 = ax.plot(n, training_accuracy[1], "-g", label="0.01 learning rate")
    rate01 = ax.plot(n, training_accuracy[2], "-b", label="0.1 learning rate")
    rate1 = ax.plot(n, training_accuracy[3], "-c", label="1 learning rate")
    rate10 = ax.plot(n, training_accuracy[4], "-m", label="10 learning rate")
    rate100= ax.plot(n, training_accuracy[5], "-k", label="100 learning rate")
    ax.legend(["learning rate=0.001", "learning rate=0.01", "learning rate=0.1",
          "learning rate=1" ,"learning rate=10","learning rate=100"])
    plt.show()


    # Visualize training loss
    fig, ax = plt.subplots()
    rate0001 = ax.plot(n, training_loss[0], "-r", label="0.001 learning rate")
    rate001 = ax.plot(n, training_loss[1], "-g", label="0.01 learning rate")
    rate01 = ax.plot(n, training_loss[2], "-b", label="0.1 learning rate")
    rate1 = ax.plot(n, training_loss[3], "-c", label="1 learning rate")
    rate10 = ax.plot(n, training_loss[4], "-m", label="10 learning rate")
    rate100= ax.plot(n, training_loss[5], "-k", label="100 learning rate")
    ax.legend(["learning rate=0.001", "learning rate=0.01", "learning rate=0.1",
               "learning rate=1" ,"learning rate=10","learning rate=100"])
    plt.show()



#Code for question 2C
def C():

    """
    Train and test a neural network with different data sizes.
    """

    # Load larger training and test data
    training_data,test_data=backprop_data.load(train_size=50000,test_size=10000)

    # Initialize a neural network with architecture [784, 40, 10]
    net=backprop_network.Network([784, 40, 10])

    # Train and test the neural network
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


# Uncomment the following lines to run the experiments
#B() #uncomment code in SGD function
#C()