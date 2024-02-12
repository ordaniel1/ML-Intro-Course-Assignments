#################################
# Your name: Or Daniel
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data



"""
    Goal: In this exercise we will implement AdaBoost and see how boosting can applied
          to real-world problems. We will focus on binary sentiment analysis,
          the task of classifying the polarity of a given text into two classes - positive and negative.
          We will use movie reviews from IMDB as our data.
          
          We will use the class of hypotheses of the form:
            h(x_i)= 1 if x_ij<=theta, -1 otherwise
            h(x_i)=-1 if x_ij<=theta,  1 otherwise
"""




np.random.seed(7)




def run_adaboost(X_train, y_train, T):
    """
    The method implements the AdaBoost algorithm

    Input:
        X_train : numpy array - The training data.
        y_train : numpy array - The labels corresponding to the training data.
        T : int - The number of iterations.



    Returns:
        hypotheses : list
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : list
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """

    l=len(X_train)
    D=np.full((l),1/l)  # Initialize weights uniformly for the first iteration
    hypotheses=[] # List to store hypotheses
    alpha_vals=[] # List to store alpha values

    # Iterate through T rounds of AdaBoost
    for t in range(T):
        # Find the best weak learner for the current iteration
        h_t=find_best_WL(X_train, y_train, D)
        hypotheses.append(h_t)

        # Calculate empirical error for the current hypothesis
        epsilon_t=calc_empirical_error(X_train, y_train,h_t, D)

        # Calculate alpha value for the current iteration
        alpha_t=0.5*np.log((1-epsilon_t)/epsilon_t)
        alpha_vals.append(alpha_t)

        # Update weights for the next iteration
        D_t_plus_1=D*(np.exp(np.array([-1*alpha_t*y_train[i]*calc_h_x(h_t, X_train[i]) for i in range(l)])))
        D=D_t_plus_1/np.sum(D_t_plus_1)

    # Return the final hypotheses and alpha values
    return hypotheses, alpha_vals




##############################################




def find_best_WL(X_train, y_train, D):

    """
    Finds the best weak learner based on ERM for decision stumps.

    Input:
        X_train : numpy array - The training data.
        y_train : numpy array - The labels corresponding to the training data.
        D : numpy array - The weights associated with each training example.

    Returns:
        Tuple -
            A tuple (h_pred, h_index, h_theta), where h_pred is the predicted value (+1 or -1),
            h_index is the feature index, and h_theta is the threshold for the best weak learner.
    """

    # Find the best weak learner for the class +1
    h_index1, h_theta1 , F1=ERM_for_Decision_Stumps(X_train, y_train, D,1)

    # Find the best weak learner for the class -1
    h_index_minus, h_theta_minus, F_minus=ERM_for_Decision_Stumps(X_train, y_train, D,-1)

    # Compare the exponential loss values and choose the weaker hypothesis
    if F1<F_minus:
        return (1, h_index1, h_theta1)
    else:
        return (-1, h_index_minus, h_theta_minus)





#ERM for decision stumps, the algorithm has taken from course book:
# "Understanding Machine Learning: From Theory to Algorithms", page 134
def ERM_for_Decision_Stumps(X_train, y_train, D,b):

    """
    Implements the ERM for Decision Stumps.

    Input:
        X_train : numpy array - The training data.
        y_train : numpy array - The labels corresponding to the training data.
        D : numpy array - The weights associated with each training example.
        b : int - The class label (+1 or -1) for which the weak learner is optimized.

    Returns:
        Tuple
            A tuple (j_star, theta, F_star), where j_star is the feature index,
            theta is the threshold, and F_star is the optimal exponential loss.
    """



    n=len(X_train)
    d=5000 # Maximum number of features to consider
    F_star=np.inf #Initialize F_star to positive infinity
    theta=0
    j_star=0

    # Iterate over each feature index j
    for j in range(d):
        # Sort the training data based on the j-th feature
        j_features=np.array([[X_train[i][j],y_train[i],D[i]] for i in range(n)])
        j_features=j_features[j_features[:,0].argsort()]

        # Add a sentinel point to the end of the sorted list
        j_features=np.vstack([j_features, [j_features[n-1][0]+1, 0, 0]])

        # Initialize the cumulative sum of weights for class b
        F=np.sum(j_features[:,2]*(j_features[:,1]==b))

        # Check for the optimal threshold and feature index
        if F<F_star:
            F_star=F
            theta=j_features[0][0] -1
            j_star=j

        # Iterate over each data point to update F and check for a better threshold
        for i in range(n):
            F-=b*j_features[i][1]*j_features[i][2]

            # Check if the current point is a candidate for a threshold
            if F<F_star and j_features[i][0]!=j_features[i+1][0]:
                F_star=F
                theta=0.5*(j_features[i][0]+j_features[i+1][0])
                j_star=j


    return j_star, theta, F_star




def calc_empirical_error(X_train, y_train,h, D):

    """
    Calculates the empirical error of a hypothesis h on the training data.

    Input:
        X_train : numpy array - The training data.
        y_train : numpy array - The labels corresponding to the training data.
        h : tuple - A tuple (h_pred, h_index, h_theta) representing the hypothesis.
        D : numpy array - The weights associated with each training example.

    Returns:
        float - The empirical error of the hypothesis h on the training data.
    """

    n=len(X_train)

    # Sum the weights of misclassified examples
    return np.sum([D[i] for i in range(n) if y_train[i]!=calc_h_x(h,X_train[i])])




def calc_h_x(h,x):

    """
    Calculates the prediction of a hypothesis h on a single example x.

    Input:
        h : tuple - A tuple (h_pred, h_index, h_theta) representing the hypothesis.
        x : numpy array - A single example from the dataset.

    Returns:
        int - h(X) - The prediction of the hypothesis h on the example x (+1 or -1).
    """

    h_pred = h[0]
    h_index = h[1]
    h_theta = h[2]

    # Make a prediction based on the hypothesis conditions
    if x[h_index]<=h_theta:
        return h_pred
    return -1*(h_pred)




def calc_error(X, Y, hypotheses, alpha_vals):

    """
    Calculate the error of the AdaBoost algorithm at each iteration.

    Input:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): True labels.
        hypotheses (list): List of hypotheses obtained during AdaBoost training.
        alpha_vals (list): List of alpha values corresponding to each hypothesis.

    Returns:
        list: List containing the error at each iteration of the AdaBoost algorithm.
    """


    n=len(X) # Number of data points
    T=len(hypotheses) # Number of iterations (hypotheses)
    prediction=np.zeros(n) # Initialize the prediction array
    result=[] # List to store errors at each iteration

    # Iterate through each hypothesis
    for t in range(T):
        # Update predictions based on the current hypothesis and its alpha value
        prediction+=np.array([np.sum([calc_h_x(hypotheses[t],X[i])*alpha_vals[t]]) for i in range(n)])

        # Calculate error at the current iteration
        error=np.sum([1 for i in range(n) if ((prediction[i]<0 and Y[i]!=-1) or(prediction[i]>=0 and Y[i]!=1))])/n

        # Store the error for the current iteration
        result.append(error)

    return result




def calc_exp_loss(X, Y, hypotheses, alpha_vals):
    """
        Calculate the exponential loss of the AdaBoost algorithm at each iteration.

        Input:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            hypotheses (list): List of hypotheses obtained during AdaBoost training.
            alpha_vals (list): List of alpha values corresponding to each hypothesis.

        Returns:
            list: List containing the exponential loss at each iteration of the AdaBoost algorithm.
        """

    m=len(X)  # Number of data points
    T=len(hypotheses) # Number of iterations (hypotheses)
    prediction = np.zeros(m) # Initialize the prediction array
    result=[] # List to store exponential loss at each iteration

    # Iterate through each hypothesis
    for t in range(T):

        # Update predictions based on the current hypothesis and its alpha value
        prediction += np.array([np.sum([calc_h_x(hypotheses[t], X[i]) * alpha_vals[t]])*-1*Y[i] for i in range(m)])

        # Store the exponential loss for the current iteration
        result.append((np.sum(np.exp(prediction)))*(1/m))

    return result


##############################################


def main():
    # Process the data and represent every review as a 5000 vector x.
    # The values of x counts the most common words in the dataset
    # (x_ij is the number of times the word w_j appears in the review r_i)
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data #vocabulary is a dictionary that maps j to w_j



    #Code for question a

    # Run AdaBoost for T=80 Iterations
    T=80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    Ts=np.arange(1,T+1,1)

    # Calculate training and test errors for each iteration
    training_errors=calc_error(X_train, y_train, hypotheses, alpha_vals)
    test_errors=calc_error(X_test, y_test, hypotheses, alpha_vals)

    # Plot the training error and the test error of the classifier corresponding to each iteration t
    fig, ax = plt.subplots()
    training = ax.plot(Ts, training_errors, "-r", label="training error")
    test = ax.plot(Ts, test_errors, "-c", label="test error")
    ax.legend(["training error", "test error"])
    plt.show()



    #Code for question b:

    # Run AdaBoost for T=10 iterations
    T=10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    # Print the selected words for each hypothesis in the final ensemble
    for h in hypotheses:
        print(vocab[h[1]])



    #Code for question c

    # Run AdaBoost for T=80 Iterations
    T=80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    # Calculate exponential loss for training and test sets over iterations
    exp_loss_train=calc_exp_loss(X_train, y_train, hypotheses, alpha_vals)
    exp_loss_test=calc_exp_loss(X_test, y_test, hypotheses, alpha_vals)

    # Plot exponential loss for training and test sets over iterations
    Ts = np.arange(1, T + 1, 1)
    fig, ax = plt.subplots()
    training = ax.plot(Ts, exp_loss_train, "-r", label="training error")
    test = ax.plot(Ts, exp_loss_test, "-c", label="test error")
    ax.legend(["exponential loss - Training", "exponential loss - Test"])
    plt.show()

if __name__ == '__main__':
    main()



