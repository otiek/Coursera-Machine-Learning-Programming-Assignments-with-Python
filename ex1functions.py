import numpy as np
import matplotlib.pyplot as plt

def plotData(x, y):
    """
    Plots the data points x and y into a new figure
    plotData(x, y) plots the data points and gives the figure axes labels of population
    and profit
    """
    plt.plot(x, y, "rx")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()

def computeCost(X, y, theta):
    """
    Computes the cost for linear regression
    J = computeCost(X, y, theta) computes the cost of using theta as the parameter
    for linear regression to fit the data points in X and y
    """
    m = y.shape[0] #number of training examples

    h = np.matmul(X, theta)
    diff = h - y.reshape(-1, 1)
    J = (1/(2*m)) * np.sum(diff**2)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta
    theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learnin rate alpha
    """
    m = y.shape[0] #number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.matmul(X, theta)
        diff = h - y.reshape(-1, 1);
        temp = np.matmul(diff.transpose(), X)
        theta = theta - (alpha/m)*temp.transpose()

        J_history[i] = computeCost(X, y, theta)
    return theta, J_history
