# Machine Learning Online class: Logistic Regression

# Load data
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop
# ==================== Part 1: Loading and Visualizing Data ====================

def loadData(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)

    x = np.zeros((numberOfLines, 2))
    y = np.zeros((numberOfLines, 1))
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFormLine = line.split(',')
        x[index, :] = listFormLine[:2]
        y[index] = listFormLine[-1]
        index += 1
    return x, y, numberOfLines

def plotData(x, y):
    f2 = plt.figure(2)

    idx_1 = np.where(y == 0)
    p1 = plt.scatter(x[idx_1, 0], x[idx_1, 1], marker='x', color='m', label='Not admitted', s=30)
    idx_2 = np.where(y == 1)
    p2 = plt.scatter(x[idx_2, 0], x[idx_2, 1], marker='+', color='c', label='Admitted', s=50)

    plt.xlabel('Exam 1 Score')
    plt.xlabel('Exam 2 Score')
    plt.legend(loc='upper right')
    # plt.show()
    return plt

# ============ Part 2: Compute Cost and Gradient ============
def sigmod(z):
    g = np.zeros(z.shape)
    # 在python 中，math.log()函数不能对矩阵直接进行操作
    # http://blog.csdn.net/u013634684/article/details/49305655
    g = 1 / (1 + np.exp(-z))


    return g

def costFunction(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    s1 = np.log(sigmod(np.dot(X, theta)))
    s2 = np.log(1 - sigmod(np.dot(X, theta)))

    s1 = s1.reshape((m, 1))
    s2 = s2.reshape((m, 1))

    s = y * s1 + (1 - y) * s2
    J = -(np.sum(s)) / m

    return J

def Gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    grad = ((X.T).dot(sigmod(np.dot(X, theta)) - y)) / m

    return grad.flatten()

def mapFeature(x1, x2):
    degree = 6
    out = np.ones(x1.shape[1])
    for i in range(degree):
        for j in range(i):
            newColumn = np.array([(x1 ** (i-j)).dot(x2 ** j)])
            np.column_stack(out, newColumn)



def plotDecisionBoundary(theta, X, y):
    f2 = plotData(X[:, 1:], y)
    # print(X[:, 1:])
    m, n = X.shape
    if n <= 3:
    # Only need 2 points to define a line, so choose two endpoints
        minVals = X[:, 1].min(0)-2
        maxVals = X[:, 1].max(0)+2
        plot_x = np.array([minVals, maxVals])
        plot_y = (-1 / theta[2]) * (plot_x.dot(theta[1]) + theta[0])
        f2.plot(plot_x, plot_y, label="Test Data", color='b')
        plt.show()

    else:
    # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j])* theta

# ============== Part 4: Predict and Accuracies ==============
def predict(theta, X):
    pass

if __name__ == '__main__':

# ==================== Part 1: Loading and Visualizing Data ====================
    x, y, numberOfLines = loadData('ex2data1.txt')

    # plotData(x, y)

# ============ Part 2: Compute Cost and Gradient ============
# 相关代码 https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
# 梯度下降法http://www.cnblogs.com/LeftNotEasy/archive/2010/12/05/mathmatic_in_machine_learning_1_regression_and_gradient_descent.html
# 雅克布矩阵http://jacoxu.com/jacobian%E7%9F%A9%E9%98%B5%E5%92%8Chessian%E7%9F%A9%E9%98%B5/

    columnOne = np.ones((numberOfLines, 1))
    X = np.column_stack((columnOne, x))

    m, n = X.shape
    # initialTheta = np.zeros((X.shape[1], 1))
    initialTheta = np.zeros(n)

    cost = costFunction(initialTheta, X, y)
    grad = Gradient(initialTheta, X, y)

    print('Cost at initial theta (zeros):\n', cost)
    print('Gradient at initial theta (zeros): \n', grad)

    print('\nProgram paused. Press enter to continue.\n')

    Result = scop.minimize(fun=costFunction, x0=initialTheta, args=(X, y), method='CG', jac=Gradient)
    optimalTheta = Result.x

    # Print theta to screen
    print('Cost at theta found by fminunc:', Result.fun)
    print('theta: \n', optimalTheta)


    # Plot Boundary
    plotDecisionBoundary(optimalTheta, X, y)

    prob = sigmod(np.array([1, 45, 85]).dot(optimalTheta))
    print('For a student with scores 45 and 85, we predict an admission probability of ', prob)