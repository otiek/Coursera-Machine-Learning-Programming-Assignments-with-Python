import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

lr = LinearRegression()
lr.fit(X, y)

theta = np.array([lr.intercept_, lr.coef_[0]])

print("Theta found by gradient descent: \n")
print("a0 = {}".format(theta[0]))
print("a1 = {}".format(theta[1]))

prediction = lr.predict(X)

plt.plot(X, y, 'rx')
plt.plot(X, prediction, '-')
plt.show()

predict1 = lr.predict(np.array(3.5).reshape(1, -1))[0]
print("For population = 35,000, we predict a profit of {}\n".format(predict1 * 10000))
predict2 = lr.predict(np.array(7).reshape(1, -1))[0]
print("For population = 70,000, predict a profit of {}\n".format(predict2 * 10000))

J = mean_squared_error(prediction, y) / 2
print("Cost of Linear Regression = {}".format(J))
