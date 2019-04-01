import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    """
    Plots the data points with + for positive examples and o for negative examples
    """
    plt.plot(X[y==0, 0], X[y==0, 1], 'yo', label='y=0')
    plt.plot(X[y==1, 0], X[y==1, 1], 'k+', label='y=1')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.xlim(-1, 1.5)
    plt.ylim(-0.8, 1.2)
    plt.legend(loc='upper right')
    plt.show()

def mapFeature(X1, X2):
    """
    Feature mapping function to polynomial features
    Returns a new array with more features, comprising of X1, X2, X1^2, X2^2,
    X1*X2, X1*X2^2,...
    """
    degree = 6
    out = np.ones((X1.shape[0], 1))

    for i in range(1, degree+1):
        for j in range(i+1):
            new = np.multiply(X1**(i-j), X2**j).reshape(-1, 1)
            out = np.append(out, new, axis=1)
    return out

def sigmoid(z):
    """"
    Returns the sigmoid of z
    """
    return 1 / (1+np.exp(-z))

def costFunctionReg(theta, X, y, lam):
    """
    Computes the cost of using theta as a parameter for regularized logistic regression
    """
    #number of training examples
    y = y.reshape(-1, 1)
    m = y.shape[0]
    #compute the cost
    theta = theta.reshape(-1, 1)
    h = sigmoid(np.matmul(X, theta))
    J = (-1/m) * np.sum(np.multiply(y, np.log(h)) + np.multiply(1-y, np.log(1-h))) + (lam/(2*m)) * np.sum(theta[1:, 0] **2)
    return J

def gradient(theta, X, y, lam):
    """
    Computes the gradient of regularized logistic regression
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
    grad[1:, 0] = grad[1:, 0] + (lam/m) * theta[1:, 0]
    return grad

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta
    Threshhold at 0.5
    """
    #number of training examples
    m = X.shape[0]
    theta = theta.reshape(-1, 1)

    temp = sigmoid(np.matmul(X, theta))
    p = (temp >= 0.5).astype(np.int)
    return p

def mapFeaturePlotting(X1, X2):
    """
    Map feature function for plotting decision boundary
    """
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out

def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary
    defined by theta
    """
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeaturePlotting(u[i], v[j]), theta)

    plt.contour(u, v, z, 0)
    plt.plot(X[y==0, 0], X[y==0, 1], 'yo', label='y=0')
    plt.plot(X[y==1, 0], X[y==1, 1], 'k+', label='y=1')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.xlim(-1, 1.5)
    plt.ylim(-0.8, 1.2)
    plt.legend(loc='upper right')
    plt.show()
