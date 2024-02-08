#################################
# Your name: Or Daniel
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py

GOAL: In this exercise we will optimize the Hinge loss (with L2 regularization) and the log loss, using
        the stochastic gradient descent implementation.
      
"""



def helper():
    """The function loads the training, validation and test sets for the digits
       0 and 8 from the MNIST data.

       Returns:
        A tuple containing the following datasets and labels:
        - train_data: Training data for digits 0 and 8.
        - train_labels: Labels for the training data.
        - validation_data: Validation data for digits 0 and 8.
        - validation_labels: Labels for the validation data.
        - test_data: Test data for digits 0 and 8.
        - test_labels: Labels for the test data.

    """
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Stochastic Gradient Descent (SGD) for hinge loss with L2 regularization.

    Input:
    - data (np.ndarray): Input data.
    - labels (np.ndarray): Labels corresponding to the input data (-1 or 1).
    - C (float): Regularization parameter.
    - eta_0 (float): Initial learning rate.
    - T (int): Number of iterations (epochs).

    Returns: The learned weight vector (np.ndarray)
    """

    # Initialize weight vector with zeros
    w=np.zeros(784)

    # Iterate over T epochs
    for t in range(T):
        # Randomly select a data point index
        i = np.random.randint(len(data))

        # Update the learning rate using the schedule eta_t = eta_0 / (t + 1)
        eta_t=eta_0/(t+1)

        # Update the weight vector based on hinge loss
        if (labels[i]*np.dot(w,data[i])<1):
            w=(1-eta_t)*w + eta_t*C*labels[i]*data[i]
        else:
            w=(1-eta_t)*w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements Stochastic Gradient Descent (SGD) for log loss.

    Input:
    - data (np.ndarray): Input data.
    - labels (np.ndarray): Labels corresponding to the input data (-1 or 1).
    - eta_0 (float): Initial learning rate.
    - T (int): Number of iterations (epochs).

    Returns: The learned weight vector (np.ndarray).
    """

    # Initialize weight vector with zeros
    w = np.zeros(784)

    # Iterate over T epochs
    for t in range(T):
        # Randomly select a data point index
        i=np.random.randint(len(data))

        # Update learning rate
        eta_t=eta_0/(t + 1)

        x=data[i]
        y=labels[i]

        # Update weight vector based on log loss
        w=w+eta_t*(softmax([0,y*np.dot(w,x)])[0])*y*x

    return w

#################################

# Place for additional code

def calc_accuracy(data, labels, w):
    """
       Calculate the accuracy of a linear classifier.

       Parameters:
       - data (np.ndarray): Input data
       - labels (np.ndarray): True labels corresponding to the input data (-1 or 1).
       - w (np.ndarray): Weight vector of the linear classifier.

       Returns:
       float: The accuracy of the classifier on the given data.
    """
    l=len(data) # Number of samples in the dataset
    prediction=np.zeros(l) # Initialize an array to store predictions
    for i in range(l): # Make predictions based on the linear classifier
        if(np.dot(w,data[i])>=0):
            y=1
        else:
            y=-1
        prediction[i]=(y==labels[i])

    # Calculate the accuracy as the average of correct predictions
    return np.average(prediction)




def run_sgd_algorithm(sgd_algorithm, **kwargs):
    """
    Run the stochastic gradient descent (SGD) algorithm with varying learning rates (etas).

    Input:
        sgd_algorithm (function): The SGD algorithm to run.
        **kwargs: Keyword arguments specific to the SGD algorithm.

    Returns:
        None: Displays a plot of average accuracy for each learning rate (eta) on a log scale.
    """
    # Define a range of eta values (learning rates) as powers of 10
    etas = [10 ** i for i in range(-5, 6)]

    # Initialize an array to store average accuracy for each eta
    average = np.zeros(11)

    # Loop over each eta value
    for j in range(11):
        # Initialize an array to store accuracy for each run
        accuracy = np.zeros(10)
        # Run the SGD algorithm 10 times for each eta
        for i in range(10):
            # Apply SGD_algorithms to obtain the weight vector
            w = sgd_algorithm(eta_0=etas[j], T=1000, **kwargs)

            # Calculate accuracy on the validation set using the obtained weight vector
            accuracy[i] = calc_accuracy(validation_data, validation_labels, w)

        # Calculate the average accuracy for the current eta
        average[j] = np.average(accuracy)

    # Find the index of the maximum average accuracy to determine the best eta
    ind = np.argmax(average)
    best_eta = etas[ind]
    # print("Best eta is ", best_eta)

    # Plot the average accuracy for each eta on a log scale
    fig, ax = plt.subplots()
    acc = ax.plot(etas, average, "-r")
    plt.xscale("log")
    plt.show()




#load data
print("load data")
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
print("data has been loaded successfully")




#Code for question 1a
run_sgd_algorithm(SGD_hinge,data=train_data, labels=train_labels, C=1)




#Code for question 1b

# Define a range of C values (regularization parameter) as powers of 10
C_values=[10**i for i in range(-5,6)]

# Initialize an array to store average accuracy for each C
average=np.zeros(11)

# Loop over each C value
for j in range(11):
    # Initialize an array to store accuracy for each run
    accuracy=np.zeros(10)
    # Run the SGD_hinge algorithm 10 times for C
    for i in range(10):
        # Apply SGD_hinge to obtain the weight vector
        w = SGD_hinge(train_data, train_labels, C_values[j], 1, 1000)

        # Calculate accuracy on the validation set using the obtained weight vector
        accuracy[i]=calc_accuracy(validation_data, validation_labels, w)

    # Calculate the average accuracy for the current C
    average[j]=np.average(accuracy)

# Find the index of the maximum average accuracy to determine the best C
ind= np.argmax(average)
best_C=C_values[ind]
#print("Best c is ", best_C)

# Plot the average accuracy for each C on a log scale
fig, ax = plt.subplots()
acc=ax.plot(C_values,average, "-r")
plt.xscale("log")
plt.show()




#Code for question 1c

# Apply the SGD_hinge algorithm with specific parameters to obtain the weight vector
# Parameters: C=0.0001 (regularization parameter), eta_0=1 (learning rate), T=20000 (number of iterations)
w = SGD_hinge(train_data, train_labels, 0.0001, 1, 20000)

# Reshape the weight vector to the shape of an image (28x28) and visualize it
plt.imshow(np.reshape(w, (28,28)), interpolation='nearest')
plt.show()




#Code for question 1d

# Calculate the accuracy of the best classifier (previously obtained weight vector) on the test dataset
accuracy=calc_accuracy(test_data, test_labels, w)

# Print the accuracy of the classifier on the test dataset
print("Accuracy of the best classifier: ", accuracy)




#Code for question 2a
run_sgd_algorithm(SGD_log,data=train_data, labels=train_labels)




#Code for question 2b

# Run SGD_log algorithm to obtain the weight vector
w = SGD_log(train_data, train_labels, 0.00001, 20000)

# Display the weight vector as an image (reshaped to 28x28)
plt.imshow(np.reshape(w, (28,28)), interpolation='nearest')
plt.show()

# Calculate accuracy on the test set using the obtained weight vector
accuracy=calc_accuracy(test_data, test_labels, w)
print("Accuracy of the best calssifier: ", accuracy)




#Code for question 2c
def SGD_log_norm(data, labels, eta_0, T):
    """
    Implements SGD for log loss and computes the L2 norm of the weight vector at each iteration.
    """
    # Initialize an array to store the L2 norm at each iteration
    norm=np.zeros(T+1)

    # Initialize the weight vector
    w = np.zeros(784)

    # Iterate over T iterations
    for t in range(T):
        # Choose a random data point
        i=np.random.randint(len(data))

        # Update the learning rate
        eta_t=eta_0/(t + 1)

        # Extract the current data point and label
        x=data[i]
        y=labels[i]

        # Update the weight vector using SGD for log loss
        w=w+eta_t*(softmax([0,y*np.dot(w,x)])[0])*y*x

        # Calculate and store the L2 norm of the weight vector at each iteration
        norm[t+1]=np.linalg.norm(w)

    #Reruen the array that stores the L2 norms
    return norm

# Generate an array of time steps for plotting
t=np.arange(0,20001,1)

# Run SGD_log_norm to obtain the L2 norm at each iteration
norm=SGD_log_norm(train_data, train_labels, 0.00001, 20000)

# Plot the L2 norm over time on a log scale
fig, ax = plt.subplots()
acc=ax.plot(t,norm, "-r")
plt.yscale("log")
plt.show()

#################################
