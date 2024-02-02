import numpy as np
import matplotlib.pyplot as plt

"""Goal: Visualizing the Hoeffding bound for Bernoulli trials
   """


# Parameters
n=20            # Number of Bernoulli trials
p=0.5           # Probability of success in each trial - Bernoulli(0.5)
N=20000         # Number of experiments

# Generate random samples representing outcomes of Bernoulli trials (1 for success, 0 for failure)
samples=np.random.binomial(1, p, (N,n));
mean=samples.mean(1) # Calculate the sample mean for each experiment

# Define a range of epsilon values (50 values of epsilon in [0,1])
epsilons=np.linspace(0,1,50)

# Calculate the empirical probability that |mean-0.5|>epsilon for each epsilon value
mean_error=abs(mean-p) #p=0.5
empirical_prob=np.zeros(50)
for i in range(50):
    empirical_prob[i]=sum(mean_error>epsilons[i])/N


# Calculate Hoeffding bound for each epsilon value
hoeffding=np.zeros(50)
for i in range(50):
    hoeffding[i]=2*np.exp(-2*n*(epsilons[i])**2)

# Print epsilon values for reference
print(epsilons)

# Plotting the empirical probability and Hoeffding bound
fig, ax = plt.subplots()
emp=ax.plot(epsilons,empirical_prob, "-r", label="Empirical Probability")
hof=ax.plot(epsilons,hoeffding, "-c", label="Hoeffding bound")
ax.legend(["Empirical Probability", "Hoeffding bound"])

# Display the plot
plt.show()