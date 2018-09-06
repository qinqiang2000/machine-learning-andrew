import numpy as np
from plotData import plotData

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt("ex2data1.txt", delimiter=',') # read comma separated data
X = data[:, 0:-1] # 读取除最后一列外的所有列
y = data[:, -1:]

m = X.shape[0]
n = X.shape[1]
## ==================== Part 1: Plotting ====================
# plotData(X, y)

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. 

#  Setup the data matrix appropriately, and add ones for the intercept term
X = np.column_stack((np.ones(m), X)) # Add a column of ones to x
