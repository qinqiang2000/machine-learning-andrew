""" 
Machine Learning Online Class
Exercise 7 | Principle Component Analysis and K-Means Clustering 
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import *

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you should complete the code in the findClosestCentroids function. 
#
print('Finding closest centroids.')

# Load an example dataset that we will be using
mat = sio.loadmat('ex7data2.mat')
X = mat['X']
# plt.figure()
# plt.plot(X[:,0], X[:,1], "rx")
# plt.show()

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print(idx[:3])
print('(the closest centroids should be 1, 3, 2 respectively)')

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.')

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids: ')
print(centroids)
print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print('\nRunning K-Means clustering on example dataset.')

K= 3
max_iters = 8

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in kMeansInitCentroids).
# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, False)
#  ============= Part 4: K-Means Clustering on Pixels ===============
#   In this exercise, you will use K-Means to compress an image. To do this,
#   you will first run K-Means on the colors of the pixels in the image and
#   then you will map each pixel onto its closest centroid.
print("\nRunning K-Means clustering on pixels from an image.")

# This creates a three-dimensional matrix A whose first two indices 
# identify a pixel position and whose last index represents red, green, or blue.
A = io.imread('bird_small.png')

print("img shape is ",A.shape)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(A)
# plt.show()

# Divide every entry in A by 255 so all values are in the range of 0 to 1
A = A / 255.

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
# X = np.reshape(A, (img_size[0] * img_size[1], 3))
X = A.reshape(-1, 3)

print('shape after reshape img:', X.shape)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly. 
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, False)

## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print('\nApplying K-Means to compress an image.')

# Find closest cluster members（runkMeans返回的idx是倒数第二迭代的，因此这里要再算一次）
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx - 1, :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(A.shape[0], A.shape[1], 3)
print('X_recovered: ', X_recovered.shape)

ax = fig.add_subplot(122)
ax.imshow(X_recovered)
plt.show()