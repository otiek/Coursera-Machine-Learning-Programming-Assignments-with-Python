import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    """
    plots the data points with + for positive examples and o for negative examples
    """
    plt.plot(X[y==0, 0], X[y==0, 1], 'yo', label='Not admitted')
    plt.plot(X[y==1, 0], X[y==1, 1], 'k+', label='Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc='upper right')
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.show()

def sigmoid(z):
    """"
    Returns the sigmoid of z
    """
    return 1 / (1+np.exp(-z))

def compute_cost(theta, X, y):
    """
    Computes the cost
    """
    #number of training examples
    y = y.reshape(-1, 1)
    m = y.shape[0]
    #compute cost and gradient
    theta = theta.reshape(-1, 1)
    h = sigmoid(np.matmul(X, theta))
    J = (-1/m) * np.sum(np.multiply(y, np.log(h)) + np.multiply(1-y, np.log(1-h)))
    return J

def compute_grad(theta, X, y):
    """
    Computes the gradient
    """
    #number of training examples
    y = y.reshape(-1, 1)
    m = y.shape[0]
    #compute the gradient
    theta = theta.reshape(-1, 1)
    h = sigmoid(np.matmul(X, theta))
    diff = h - y
    temp = np.matmul(X.transpose(), diff)
    grad = 1/m * temp
    return grad.flatten()

def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary
    defined by theta
    """
    x = np.arange(30, 100, 0.1)
    boundary = -1/theta[2] * (theta[0] + theta[1] * x)
    plt.plot(x, boundary, '-')
    plt.plot(X[y==0, 1], X[y==0, 2], 'yo', label='Not admitted')
    plt.plot(X[y==1, 1], X[y==1, 2], 'k+', label='Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc='upper right')
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.show()
