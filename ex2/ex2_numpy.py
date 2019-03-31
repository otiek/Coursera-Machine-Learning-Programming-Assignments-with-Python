import numpy as np
import matplotlib.pyplot as plt
from ex2functions import *
import scipy.optimize as opt

#Load Data
#First 2 columns are the exam scores and the 3rd column contains the label
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

#Plot the data
plotData(X, y)

#Compute the cost and gradient for logistic regression
#Add the intercept term to X
[m, n] = X.shape
ones = np.ones((m, 1))
X = np.concatenate((ones, X), axis=1)

#Inititalize theta to zeros
theta = np.zeros(n+1)

cost = compute_cost(theta, X, y)
grad = compute_grad(theta, X, y)
print("\nCost at initial theta (zeros): {}".format(cost))
print("Expected cost (approx): 0.693\n")
print("Gradient at initial theta (zeros):")
for i in range(grad.shape[0]):
    print(grad[i])
print("Expected gradients (approx):\n-0.1000\n-12.0092\n-11.2628\n")

#Compute the cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = compute_cost(test_theta, X, y)
grad = compute_grad(test_theta, X, y)
print("\nCost at test theta: {}".format(cost))
print("Expected cost (approx): 0.218\n")
print("Gradient at test theta:")
for i in range(grad.shape[0]):
    print(grad[i])
print("Expected gradients (approx):\n0.043\n2.566\n2.647\n")

#Use SciPy's optimization function fmin_tnc to optimize the parameters
result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=compute_grad, args=(X, y), disp=0)
theta = result[0]
cost = compute_cost(theta, X, y)
print("Cost at theta found by fmin_tns: {}".format(cost))
print("Expected cost (approx): 0.203\n")
print("Theta:")
for i in range(theta.shape[0]):
    print(theta[i])
print("Expected theta :\n-25.161\n0.206\n0.201\n")

#Plot the decision boundary
plotDecisionBoundary(theta, X, y)
