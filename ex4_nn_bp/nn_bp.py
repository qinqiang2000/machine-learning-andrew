## Machine Learning Online Class - Exercise 4 Neural Network Learning
#
import numpy as np
import scipy.io as sio   # Used to load the OCTAVE *.mat files
import scipy.optimize as op
import random   #To pick random images to display
from displayData import displayData
from unrollutility import unrollParams
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predictNN import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)
img_width = 20

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#
mat = sio.loadmat('ex4data1.mat')
X, y = mat['X'], mat['y']
m = X.shape[0]

# 样本中，0用10替代了(MATLAB数组从1开始的原因)
print(np.unique(y), "y.shape=", y.shape)

# Randomly select 100 data points to display
sel_indics = random.sample(range(m), 20)
sel = X[sel_indics, :]

# for idx, val in enumerate(y[sel_indics]):
#   print(idx, " ", val)

# displayData(sel, img_width)

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

# Load the weights into variables Theta1 and Theta2
mat2 = sio.loadmat('ex4weights.mat')
Theta1, Theta2 = mat2['Theta1'], mat2['Theta2']
print("Theta1 has shape:",Theta1.shape, Theta1[0,0], Theta1[-1,-1])
print("Theta2 has shape:",Theta2.shape, Theta2[0,0], Theta2[-1,-1])

#  Unroll parameters 
nn_params = unrollParams([Theta1, Theta2])

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lamda = 0

J, Grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda)

print("Cost at parameters (loaded from ex4weights): ", J, "(this value should be about 0.287629)")

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#      

# Weight regularization parameter (we set this to 1 here).
lamda = 1

J, Grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda)

print("Cost(regularized) parameters (loaded from ex4weights): ", J, "(this value should be about 0.383770)")

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.py file.
#
g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('\nSigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:  ', g)

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.py)
print('\nInitializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = unrollParams([initial_Theta1, initial_Theta2])

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#

print('\nChecking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#
print('\nChecking Backpropagation (w/ Regularization) ... ')

#  Check gradients by running checkNNGradients
lamda = 3
checkNNGradients(lamda)

# Also output the costFunction debugging values
debug_J, debug_grad = nnCostFunction(nn_params, input_layer_size, 
                          hidden_layer_size, num_labels, X, y, lamda);

print("Cost at (fixed) debugging parameters (w/ lambda = %f): %f \n" % (lamda, debug_J), 
         '(for lambda = 3, this value should be about 0.576051)');

## =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... ')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
minimize_method = "L-BFGS-B" # CG, L-BFGS-B, TNC, ...

opts = {'maxiter':100}
if minimize_method == "TNC":
    opts['maxCGit'] = 0
    opts['stepmx'] = 500
elif minimize_method == 'L-BFGS-B':
    opts['eps'] = 1e-8
else:
    pass

#  You should also try different values of lambda
lamda = 1;

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
Result = op.minimize(fun = nnCostFunction,
                     x0 = initial_nn_params,
                     args = (input_layer_size, hidden_layer_size, num_labels, X, y, lamda),
                     method = minimize_method,
                     jac = True,
                     options= opts)
 
optimal_theta = Result.x;

# Obtain Theta1 and Theta2 back from optimal_theta
Theta1 = optimal_theta[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
Theta2 = optimal_theta[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

print("Theta1 has shape:",Theta1.shape, Theta1[0,0], Theta1[-1,-1])
print("Theta2 has shape:",Theta2.shape, Theta2[0,0], Theta2[-1,-1])
## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... ')
# displayData(Theta1[:, 1:])

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: ',  np.mean(y.ravel() == pred) * 100)