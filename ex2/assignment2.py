#################################
# Your name: Or Daniel
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.

    Goal: Study the hypothesis class of a finite union of disjoint intervals, and the properties of the ERM algorithm
          for this class.
          Let the sample space be X=[0,1] and assume we study a binary classification problem.
          We will try to learn using an hypothesis class H that consists of k disjoint intervals.
          For each such k disjoint intervals, we define the corresponding hypothesis h as:
          if x in the union of intervals h(x)=1, otherwise: h(x)=0
    """




    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # Generate sorted random samples from a uniform distribution
        X=np.sort(np.random.uniform(size=m))

        # Define boolean conditions based on the ranges specified
        X_bool=(X<=0.2) | (np.logical_and((0.4<=X), (X<=0.6))) | (X>=0.8)

        # Generate corresponding Y values based on the boolean conditions
        Y = np.array([np.random.choice([1, 0], p=[0.8, 0.2], size=1) if x
                      else np.random.choice([1, 0], p=[0.1, 0.9], size=1) for x in X_bool])

        # Stack X and Y arrays vertically and transpose the result
        return np.vstack((X,Y.T)).T





    def calculate_true_error(self, I):
        """
            The function calculates the true error for a given set of intervals.

            Input:  I (list) -  A list of intervals.

            Returns: The calculated true error (float).
            """

        # Define real intervals and their complements
        real_intervals=[(0,0.2),(0.4,0.6),(0.8,1)]
        real_complement=[(0.2,0.4),(0.6,0.8)]

        #Calculate the complement intervals of I
        I_complement=[]

        # Handle the case where the first interval starts from a value greater than 0
        if (I[0][0] != 0):
            I_complement.append((0,I[0][0]))
        # Calculate the complement intervals
        for i in range(len(I)-1):
            I_complement.append((I[i][1],I[i+1][0]))
        # Handle the case where the last interval ends at a value less than 1
        if (I[len(I)-1][1]!=1):
            I_complement.append((I[len(I)-1][1],1))

        #Calculate the true error
        error = 0
        temp=0

        #True error = ep(h) = P[h(X)!=Y] = P[h(X)!=Y|h(X)=0] + P[h(X)!=Y|h(X)=1] =
        #      = P[Y=1|h(X)=0] + P[Y=0|h(X)=1]

        # Calculate error for h(x) = 0 case
        for i in I_complement:

            # Calculate the contribution of intervals in real_intervals (where Y=1) to the error
            for j in real_intervals:
                temp+=self.size_of_intersect(i,j)
            temp*=0.8
            error+=temp
            temp=0

            # Calculate the contribution of intervals in real_complement (where Y=0) to the error
            for k in real_complement:
                temp+=self.size_of_intersect(i,k)
            temp*=0.1
            error+=temp
            temp=0

        # Calculate error for h(x) = 1 case
        for i in I:

            # Calculate the contribution of intervals in real_intervals (where Y=1) to the error
            for j in real_intervals:
                temp+=self.size_of_intersect(i,j)
            temp*=0.2
            error+=temp
            temp=0

            # Calculate the contribution of intervals in real_complement (where Y=0) to the error
            for k in real_complement:
                temp+=self.size_of_intersect(i,k)
            temp*=0.9
            error+=temp
            temp=0
        return error





    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """

        # Calculate the number of steps based on the provided range and step size
        size=int((m_last-m_first)/step+1)

        # Initialize arrays to store empirical and true errors for each m
        empirical_error=np.zeros(size)
        true_error=np.zeros(size)

        # Iterate over the range of m values
        for i in range(size):
            m=m_first+(step*i)

            # Initialize arrays to store errors for each iteration of the experiment
            m_empirical_error=np.zeros(T)
            m_true_error = np.zeros(T)

            # Run the experiment T times
            for j in range(T):
                # Generate a data sample
                sample=self.sample_from_D(m)

                # Extract X and Y from the sample
                xs=sample[:,0]
                ys=sample[:,1]

                # Find the best intervals using the ERM algorithm
                list_of_intervals,best_error=intervals.find_best_interval(xs,ys,k)

                # Calculate empirical error and true error for the current experiment iteration
                m_empirical_error[j]=best_error/m #find_best_interval returns the error count
                m_true_error[j]=self.calculate_true_error(list_of_intervals)

            # Calculate the average empirical and true errors for the current m
            empirical_error[i]=np.average(m_empirical_error)
            true_error[i]=np.average(m_true_error)

        # Generate an array of m values for plotting
        n = np.arange(m_first, m_last+1, step)

        # Plot the average empirical and true errors
        fig, ax = plt.subplots()
        plt.ylim([0, 0.5])
        emp = ax.plot(n, empirical_error, "-r", label="empirical error")
        true = ax.plot(n, true_error, "-c", label="true error")
        ax.legend(["empirical error", "true error"])
        plt.show()

        # Return a two-dimensional array containing the average empirical and true errors
        return np.vstack((empirical_error, true_error.T)).T





    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        # Calculate the number of steps based on the provided range and step size
        size=int((k_last-k_first)/step + 1)

        # Initialize arrays to store empirical and true errors for each k
        empirical_error = np.zeros(size)
        true_error = np.zeros(size)

        # Generate a data sample
        sample=self.sample_from_D(m)

        # Extract X and Y from the sample
        xs = sample[:, 0]
        ys = sample[:, 1]

        # Iterate over the range of k values
        for i in range(size):
            k=k_first+(step*i)

            # Find the best intervals using the ERM algorithm
            list_of_intervals, best_error = intervals.find_best_interval(xs, ys, k)

            # Calculate empirical error and true error for the current k
            empirical_error[i]=best_error/m
            true_error[i]=self.calculate_true_error(list_of_intervals)

        # Generate an array of k values for plotting
        n = np.arange(k_first, k_last + 1, step)

        # Plot the empirical and true errors as a function of k
        fig, ax = plt.subplots()
        plt.ylim([0, 0.5])
        emp = ax.plot(n, empirical_error, "-r", label="empirical error")
        true = ax.plot(n, true_error, "-c", label="true error")
        ax.legend(["empirical error", "true error"])
        plt.show()

        # Return the k value that minimizes the empirical error
        return np.argmin(empirical_error)*step+k_first





    def experiment_k_range_srm(self, m, k_first, k_last, step):

        """The function use the principle of SRM (structural risk minimization) to search for
        a k that gives a good test error.
        Plots additionally the penalty for the best ERM hypothesis
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """

        # Calculate the number of steps based on the provided range and step size
        size = int((k_last - k_first) / step + 1)

        # Initialize arrays to store empirical error, true error, and penalty for each k
        empirical_error = np.zeros(size)
        true_error = np.zeros(size)
        #penalty function is 2sqrt( (VCdim(Hk)+ln(2/delta))/m )
        #VCdim(Hk)=2k, delta=0.1
        penalty=np.array([2*np.sqrt((2*k+np.log(2/0.1))/m) for k in range(k_first,k_last+1,step)])

        # Generate a data sample
        sample = self.sample_from_D(m)

        # Extract X and Y from the sample
        xs = sample[:, 0]
        ys = sample[:, 1]

        # Iterate over the range of k values
        for i in range(size):
            k = k_first + (step * i)

            # Find the best intervals using the ERM algorithm
            list_of_intervals, best_error = intervals.find_best_interval(xs, ys, k)

            # Calculate empirical error and true error for the current k
            empirical_error[i] = best_error / m
            true_error[i] = self.calculate_true_error(list_of_intervals)

        # Calculate the sum of penalty and empirical error
        pen_emp_sum=empirical_error+penalty

        # Generate an array of k values for plotting
        n = np.arange(k_first, k_last + 1, step)

        # Plot the empirical error, true error, penalty, and the sum of penalty and empirical error as a function of k
        fig, ax = plt.subplots()
        plt.ylim([0, 0.5])
        emp = ax.plot(n, empirical_error, "-r", label="empirical error")
        true = ax.plot(n, true_error, "-c", label="true error")
        pen = ax.plot(n, penalty, "-g", label="penalty")
        emp_pen = ax.plot(n, pen_emp_sum, "-m", label="sum of penalty and empirical error")
        ax.legend(["empirical error", "true error", "penalty","sum of penalty and empirical error"])
        plt.show()

        # Return the k value that minimizes the sum of penalty and empirical error
        return np.argmin(pen_emp_sum) * step + k_first





    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        # Generate a data sample and shuffle it
        sample=self.sample_from_D(m)
        np.random.shuffle(sample)

        # Split the shuffled sample into training (80%) and test (20%) sets
        sample1=sample[:int(0.8*m)]
        sample2=sample[int(0.8*m):]

        s1=sample1[np.argsort(sample1[:,0])] #training set
        s2=sample2[np.argsort(sample2[:,0])] #test set

        # Extract X and Y values from the training and test sets
        xs1=s1[:, 0] #data for training set
        ys1=s1[:, 1] #labels for training set
        xs2=s2[:, 0] #data for test set
        ys2=s2[:, 1] #labels for test set

        # Initialize an array to store test errors for different values of k
        error_s2=np.zeros(10)
        n = int(m * 0.2)

        # Iterate over different values of k and calculate test errors
        for k in range(1,11):

            # Find the best intervals using the ERM algorithm
            I, best_error = intervals.find_best_interval(xs1, ys1, k)

            #Calculate test error
            error_s2[k-1]=self.calculate_test_error(I,xs2,ys2,n)

        # Return the best k value that minimizes test error
        return np.argmin(error_s2)+1



    #################################
    # Place for additional methods



    def size_of_intersect(self, A, B):
        """
            Calculate the size of the intersection between two intervals.

            Input:  A (tuple): First interval represented as a tuple (start, end).
                        B (tuple): Second interval represented as a tuple (start, end).

            Returns: Size of the intersection between the two intervals (float)
        """
        maxOpen=max(A[0],B[0])
        minClose=min(A[1], B[1])
        if(maxOpen>minClose):
            return 0
        else:
            return minClose-maxOpen





    def calculate_test_error(self,I, xs, ys, n):
        """Calculate the test error for a given hypothesis.

        Input:
            I - a list of intervals defining the hypothesis.
            xs - array of X values in the test set.
            ys - array of labels in the test set.
            n - an integer, the size of the test set.

        Returns:
            float: The test error, calculated as the proportion of misclassified instances.
        """

        # Initialize the error count
        error=0

        # Iterate over each data point in the test set
        for j in range(n):
            label = 0
            # Check if the data point falls within any interval of the hypothesis
            for i in I:
                if (i[0]<=xs[j]<=i[1]):
                    label=1
                    break

            # Check if the predicted label matches the true label
            if (label!=ys[j]):
                error+=1

        # Calculate and return the test error
        return error/n
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)



