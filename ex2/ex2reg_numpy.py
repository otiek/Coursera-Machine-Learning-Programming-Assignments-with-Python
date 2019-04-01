import numpy as np
import matplotlib.pyplot as plt
from ex2regfunctions import *
import scipy.optimize as opt

#Load and plot data
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
plotData(X, y)

#Add polynomial features up to the sixth power
X = mapFeature(X[:, 0], X[:, 1])

#Inititalize fitting parameters
initial_theta = np.zeros(X.shape[1])

#Set regularization parameter lambda to 1
lam = 1

cost = costFunctionReg(initial_theta, X, y, lam)
grad = gradient(initial_theta, X, y, lam)

print("Cost at initital theta (zeros): {}".format(cost))
print("Expected cost (approx): 0.693\n")
print("Gradient at initial theta (zeros) - first five values only:")
for i in range(5):
    print(grad[i, 0])
print("Expected gradients (approx) - first five values only:\n0.0085\n0.0188\n0.0001\n0.0503\n0.0115\n")

#Compute cost for theta with all ones and lambda=10
test_theta = np.ones(X.shape[1])
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradient(test_theta, X, y, 10)

print("Cost at test theta (lambda=10): {}".format(cost))
print("Expected cost (approx): 3.16\n")
print("Gradient at test theta - first five valyes only:")
for i in range(5):
    print(grad[i, 0])
print("Expected gradients (approx) - first five values only:\n0.3460\n0.1614\n0.1948\n0.2269\n0.0922\n")
#Learng parameters using SciPy's fmin_tnc function
result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, fprime=gradient, args=(X, y, lam), disp=0)
theta = result[0]

#Predict the label of training data with parameters and calculate accuracy
p = predict(theta, X)
y = y.reshape(-1, 1)
accuracy = np.count_nonzero(p == y) / y.shape[0] * 100
print("Train accuracy: {}".format(accuracy))
print("Expected accuracy (with lambda =1): 83.1 (approx)")

#plot decision boundary
plotDecisionBoundary(theta, data[:, :2], data[:, 2])
