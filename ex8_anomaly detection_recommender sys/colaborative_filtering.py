import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
from unrollutility import unrollParams
from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from normalizeRatings import normalizeRatings

import codecs


""" 
=============== Part 1: Loading movie ratings dataset ================
 You will start by loading the movie ratings dataset to understand the
 structure of the data.
 """ 
print("Loading movie ratings dataset")

# Load data
mat = sio.loadmat('ex8_movies.mat')

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i
Y, R = mat['Y'], mat['R']
nm, nu = Y.shape

#  From the matrix, we can compute statistics like average rating.
print("Average rating for movie 1 (Toy Story): %f / 5" %
    np.mean(Y[0, np.where(R[0,:] == 1)]) )

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.ylabel('Movies (%d)' % nm, fontsize=20)
plt.xlabel('Users (%d)' % nu, fontsize=20)
# plt.show()

""" 
============ Part 2: Collaborative Filtering Cost Function ===========
 You will now implement the cost function for collaborative filtering.
 To help you debug your cost function, we have included set of weights
 that we trained on that. Specifically, you should complete the code in 
 cofiCostFunc.m to return J. 
"""
print("\nEvaluate cost function.")

# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
mat = sio.loadmat ('ex8_movieParams.mat')
X = mat['X']
Theta = mat['Theta']
num_users = int(mat['num_users'])
num_movies = int(mat['num_movies'])
num_features = int(mat['num_features'])

#  Reduce the data set size so that this runs faster
num_users = 4 
num_movies = 5 
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

# Evaluate cost function
J, grad = cofiCostFunc(unrollParams([X, Theta]), Y, R, num_users, num_movies
             , num_features, 0)
print('Cost at loaded parameters: %f '%J, '\n(this value should be about 22.22)')

J, grad = cofiCostFunc(unrollParams([X, Theta]), Y, R, num_users, num_movies
             , num_features, 1.5)
print('Cost at loaded parameters(lambad={}):{}'.format(1.5, J) , '\n(this value should be about 31.34)')

""" 
============== Part 3: Collaborative Filtering Gradient ==============
 Once your cost function matches up with ours, you should now implement 
 the collaborative filtering gradient function. Specifically, you should 
 complete the code in cofiCostFunc.m to return the grad argument.
 """ 
print('\nChecking Gradients ... ')

# Check gradients by running checkNNGradients
checkCostFunction(unrollParams([X, Theta]), Y, R, num_users, num_movies, num_features)
checkCostFunction(unrollParams([X, Theta]), Y, R, num_users, num_movies, num_features, 1.5)

"""  注：part 4,5已经在上面做了 """
""" 
============== Part 6: Entering ratings for a new user ===============
 Before we will train the collaborative filtering model, we will first
 add ratings that correspond to a new user that we just observed. This
 part of the code will also allow you to put in your own ratings for the
 movies in our dataset!
 """
print("\n New user rattings:")

# So, this file has the list of movies and their respective index in the Y vector
# Let's make a list of strings to reference later
movieList = []
with codecs.open("movie_ids.txt", 'r', encoding = "ISO-8859-1") as f:
    for line in f:
        movieList.append(' '.join(line.strip('\n').split(' ')[1:]))

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated %d for %s" % (my_ratings[i], movieList[i]))

""" 
================== Part 7: Learning Movie Ratings ====================
 Now, you will train the collaborative filtering model on a movie rating 
 dataset of 1682 movies and 943 users
"""
mat = sio.loadmat ('ex8_movies.mat')

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i
Y = mat['Y']
R = mat['R']

#  Add our own ratings to the data matrix
Y = np.c_[Y, my_ratings] # (1682, 944)
R = np.c_[R, my_ratings != 0] # (1682, 944)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

# Set Initial Parameters (Theta, X)
X = np.random.random((num_movies, num_features))
Theta = np.random.random((num_users, num_features))

initial_parameters = unrollParams([X, Theta])

# Set Regularization
l = 10

# Now, costFunction is a function that takes in only one argument
minimize_method = "CG" # CG, L-BFGS-B, TNC, ...
opts = {'maxiter':100}
if minimize_method == 'L-BFGS-B':
    opts['eps'] = 1e-8
else:
    pass
res = opt.minimize(fun=cofiCostFunc,
                   x0=initial_parameters,
                   args=(Y, R, num_users, num_movies, num_features, l),
                   method=minimize_method,
                   jac=True,
                   options=opts)
ret = res.x

# unfold X and Theta matrices    
X = ret[:num_movies*num_features].reshape(num_movies, num_features)
Theta = ret[num_movies*num_features:].reshape(num_users, num_features)

print('\nRecommender system learning completed.')

""" 
================== Part 8: Recommendation for you ====================
 After training the model, you can now make recommendations by computing
 the predictions matrix. 
"""
# 所有用户的分数矩阵
p = X @ Theta.T

# 最后一个用户的预测分数， 也就是我们刚才添加的用户
my_predictions = p[:,-1] + Ymean.flatten()

ix = np.argsort(-my_predictions)
print("\n Top recommendations for you:")
for i in range(10):
    j = ix[i]
    print("Predicting rating %.1f for movie %s" 
        % (my_predictions[j], movieList[j]))    