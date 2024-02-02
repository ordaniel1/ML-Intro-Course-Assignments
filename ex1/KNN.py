import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


#Nearest Neighbor - QUESTION 2

"""Goal: Study the performance of the Nearest Neighbor algorithm on the MNIST dataset.
   The MNIST dataset consist of images of handwritten digits, along with their labels.
   Each image has 28x28 pixels, where each pixel is in gray-scale, and can get an integer
   value from 0 to 255. Each label is a digit between 0 and 9. The dataset has 70000 images.
   We will treat each picture as a vector of size 784.
   """


#Initialization
#load the dataset with sklearn
mnist = fetch_openml('mnist_784', as_frame=False)
data=mnist['data']
labels=mnist['target']

#define the training and test sets of images
idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]





# Code for question 2a
"""The function accepts as input: 
   (1) a set of train images
   (2) a vector of labels, corresponding to the images
   (3) a query image
   (4) a number k - Number of neighbors to consider in the KNN algorithm
   
   The function implements the KNN algorithm to return a prediction of the query,
   given the train  images and labels.
   The function uses the k nearest neighbors, using the Euclidean L2 metric. 
   In case of a tie between the k labels of neighbors, it chooses an arbitrary option
   """
def KNN(train_images, labels, query_image, k):
    # Calculate the Euclidean distances between the query image and all train images
    distance=np.array([np.linalg.norm(query_image-x) for x in train_images])

    # Sort indices of train images in ascending order based on distances
    sorted_indices=np.argsort(distance)

    # Select the labels of the k closest train images
    sorted_labels=np.array([labels[sorted_indices[i]] for i in range(k)])

    # Count the occurrences of each unique label among the k neighbors
    values, counts = np.unique(sorted_labels, return_counts=True)

    # Find the index of the most frequent label
    ind = np.argmax(counts)

    # Return the label with the highest occurrence (arbitrary choice if tie)
    return values[ind]





# Code for question 2b
"""The function accepts as input: 
   (1) a set of train images
   (2) a vector of labels, corresponding to the images
   (3) a query image
   (4) a number k - Number of neighbors to consider in the KNN algorithm
   (5) a number n - Number of training samples to use (a subset of the full training set).
   
   The function iterates over 1000 query images, using the KNN function to predict the label, 
   based on the provided training set.
   The function returns the accuracy - it's calculated by counting the number of correct predictions.
   """
def prediction(train_images, labels, query_images, k,n):
    # Initialize an array to store predictions for each query image
    prediction = np.zeros(1000)

    # Loop over each query image
    for i in range(1000):
        # Use the KNN algorithm to predict the label for the current query image
        prediction[i] = KNN(train_images[:n], labels[:n], query_images[i], k)

    # Convert the labels of the test set to float for comparison
    real_labels = test_labels.astype(float)

    # Calculate the accuracy by comparing predicted labels with actual labels
    accuracy = (sum(prediction == real_labels)) / 1000

    # Return the computed accuracy
    return accuracy







# Code for question 2c

# Create an array 'K' ranging from 1 to 100 (inclusive)
K=np.arange(1,101)

# Initialize an array 'accuracy' to store accuracy values for each value of k
accuracy=np.zeros(100)

# Iterate through values of k (1 to 100)
for i in range(1,101):
    # Calculate accuracy for the KNN algorithm with k=i and n=1000
    accuracy[i-1]=prediction(train, train_labels, test, i, 1000)

#Print the value of k that gives the highest accuracy
print(np.argmax(accuracy)+1)

# Plotting the accuracy values against different values of k
fig, ax = plt.subplots()
acc=ax.plot(K,accuracy, "-r")
# Display the plot
plt.show()






# Code for question 2d
# Create an array 'n' ranging from 100 to 5000 with a step of 100
n=np.arange(100,5001,100);

# Initialize an array 'accuracy' to store accuracy values for each value of n
accuracy=np.zeros(50)

# Iterate through values of n (100, 200, ..., 5000)
for i in range(100,5001,100):
    # Calculate accuracy for the KNN algorithm with k=1 and varying n
    t=int(i/100)-1
    accuracy[t]=prediction(train, train_labels, test, 1, i)

# Plotting the accuracy values against different values of n
fig, ax = plt.subplots()
acc=ax.plot(n,accuracy, "-r")
# Display the plot
plt.show()
