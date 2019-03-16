import numpy as np
import matplotlib.pyplot as plt
from ex1functions import *

print("Loading data ...\n")
data = np.loadtxt("ex1data1.txt", delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = y.shape[0]

plotData(X, y)

input("Press Enter to continue...")

ones = np.ones((m, 1))
X = np.concatenate((ones, data[:,0].reshape(-1, 1)), axis=1) #Add a column of ones to x
theta = np.zeros((2, 1)) #initialize fitting parameters

#Some gradient descent settings
iterations = 1500
alpha = 0.01

print("Testing the cost function ...\n")
#compute and display the initial cost
J = computeCost(X, y, theta)
print("With theta = [0,0]\nCost computed = {}\n".format(J))
print("Expected cost value (approx) 32.07 \n")

#further testing of the cost ex1functions
J = computeCost(X, y, np.array([-1, 2]).reshape(-1, 1))
print("With theta = [-1,2]\nCost computed = {}\n".format(J))
print("Expected cost value (approx) 54.24 \n")
input("Press Enter to continue...")

print("\nRunning Gradient Descent ...\n")
#run gradient descent
theta, J = gradientDescent(X, y, theta, alpha, iterations)

#print theta to screen
print("Theta found by gradient descent: \n")
print("{}\n{}".format(theta[0, 0], theta[1, 0]))
print("Expected theta values (approx)\n")
print("-3.6303\n 1.1664\n")

#Plot the linear fit
plt.plot(X[:, 1], y, "rx")
plt.plot(X[:, 1], np.matmul(X, theta), '-')
plt.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul(np.array([1, 3.5]).reshape(1, -1), theta)
print("For population = 35,000, we predict a profit of {}\n".format(predict1[0, 0] * 10000))
predict2 = np.matmul(np.array([1, 7]).reshape(1, -1), theta)
print("For population = 70,000, predict a profit of {}\n".format(predict2[0, 0] * 10000))

print("Cost of Linear Regression = {}".format(J[-1, 0]))
