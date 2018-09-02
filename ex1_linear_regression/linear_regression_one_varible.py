import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("ex1data1.txt", delimiter=',') # read comma separated data
X = data[:, 0] # 向量
y = data[:, 1]
m = y.size # number of training examples
X = X.reshape(m, 1) # 向量转矩阵
y = y.reshape(m, 1)

# Plot the data to see what it looks like
# plt.plot(X, y, 'rx', markersize=10)
# plt.ylabel('Profit in $10,000s')
# plt.xlabel('Population of City in 10,000s')
# plt.show()

# =================== Cost and Gradient descent ===================
X = np.column_stack((np.ones(m), X)) # Add a column of ones to x
theta = np.zeros((2,1)) # initialize fitting parameters
iterations = 1500
alpha = 0.01

# Compute cost for linear regression
# J = computeCost(X, y, theta) computes the cost of using theta as the
# parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
    m = y.size  # number of training examples
    J = 0
    return ((X.dot(theta) - y) ** 2).sum() / (2 * m)

# Performs gradient descent to learn theta
# theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
# taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for iter in range(0, num_iters):
        theta = theta - X.T.dot(X.dot(theta) - y) * alpha / m

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history

theta,J_history = gradientDescent(X, y, theta, alpha, iterations)

#predict 预测一下
predict1 = np.array([[1,3.5]]).dot(theta)
predict2 = np.array([[1,7]]).dot(theta)
print(predict1 * 10000, predict2 * 10000)

#plot the result可视化一下回归的曲线#plot t 
plt.subplot(211)
plt.scatter(X[:,1], y, color = 'r', marker= 'x')
plt.xlabel('X')
plt.ylabel('y')

plt.plot(X[:,1], X.dot(theta), '-', color = 'blue')
#可视化一下cost变化曲线
plt.subplot(212)
plt.plot(J_history)
plt.xlabel('iters')
plt.ylabel('cost')
# plt.show()

# =============  Visualizing J(theta_0, theta_1) =============
# Grid over which we will calculate J
size = 100
theta0_vals = np.linspace(-10, 10, size);
theta1_vals = np.linspace(-1, 4, size);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size));

# Fill out J_vals
for i in range(size):
	for j in range(size):
		t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
		# t = np.array([[theta0_vals[i]], [theta1_vals[j]]]).reshape(-1,1)
		J_vals[i, j] = computeCost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta)$')
plt.show()


# 绘制轮廓曲线,因为J与theta0和theta1两个参数有关#绘制轮廓曲线, 
contourFig = plt.figure()
ax = contourFig.add_subplot(111)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')

CS = ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2,3,20))
plt.clabel(CS, inline=1, fontsize=10)

# 绘制最优解
ax.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
plt.show()
