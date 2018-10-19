# Exercise 7 | Principle Component Analysis and K-Means Clustering

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from pca_fun import *
from projectData import projectData
from recoverData import recoverData
from displayData import displayData

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.')

mat = sio.loadmat('ex7data1.mat')
X = mat['X']

plt.plot(X[:,0], X[:,1], 'o', mfc='none', mec='b')

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset.')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S, V = pca(X_norm)
print(U.shape, S.shape, V.shape)

print('Top eigenvector: ')
print(U[:,0])
print('(you should expect to see -0.707107 -0.707107)')

plt.plot([mu[0], mu[0] + 1.5*S[0]*U[0,0]], 
         [mu[1], mu[1] + 1.5*S[0]*U[0,1]],
        c='r', linewidth=2, label='1st Principal Component')
plt.plot([mu[0], mu[0] + 1.5*S[1]*U[1,0]], 
         [mu[1], mu[1] + 1.5*S[1]*U[1,1]],
        c='y', linewidth=2, label='2nd Principal Component')

plt.axis("equal")
plt.legend()
# plt.show()

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
print("\nDimension reduction on example dataset.")

#  Plot the normalized dataset (returned from pca)
plt.figure()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'o' , mfc='none', mec='b')
plt.axis([-4, 3, -4, 3])

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f' % Z[0])
print('(this value should be about 1.481274)')

X_rec  = recoverData(Z, U, K);
print('\nApproximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')

#  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'o' , mfc='none', mec='r')
for i in range(X_norm.shape[0]):
    plt.plot([X_norm[i,0],X_rec[i,0]],[X_norm[i,1],X_rec[i,1]], 'k--', linewidth=1)

# plt.show()

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset.')

#  Load Face dataset(5K图片，用5000 * 1024存储, 每张图片1024=32*32)
mat = sio.loadmat('ex7faces.mat')
X = mat['X']

# Display the first 100 faces in the dataset
displayData(X)

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print("\nRunning PCA on face datase.t(this might take a minute or two ...)")

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S, V = pca(X_norm)
print("U.shape:", U.shape, "S.shape:", S.shape, "V.shape:", V.shape)

displayData(U[:,:36].T, 6, 6)

# plt.show()

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('\nDimension reduction for face dataset')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ', Z.shape)

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print("\nVisualizing the projected (reduced dimension) faces.")

X_rec  = recoverData(Z, U, K)
print(X_rec.shape)

displayData(X_rec)
plt.show()