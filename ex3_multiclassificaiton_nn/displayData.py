import numpy as np
import math
import scipy.misc as smisc # Used to show matrix as an image
import matplotlib.pyplot as plt

#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the 
#   displayed array if requested.
def displayData(X, example_width):
  m, n = X.shape
  example_height = int(n / example_width)

  # Compute number of items to display
  display_rows = math.floor(math.sqrt(m));
  display_cols = int(math.ceil(m / display_rows));

  big_picture = np.zeros((example_height * display_rows
                    , example_width * display_cols))

  # Copy each example into a patch on the display array
  curr_ex = 0;
  
  for i in range(display_rows):
    for j in range(display_cols):
      if curr_ex >= m:
        break;

      simg = X[curr_ex].reshape(example_width, example_height).T
      ih = i * example_height
      jw = j * example_width
      big_picture[ih : ih + example_height, jw : jw + example_width] = simg

      curr_ex += 1

    if curr_ex >= m:
        break;
  
  fig = plt.figure(figsize=(6,6))
  img = smisc.toimage(big_picture)
  plt.imshow(img, cmap = plt.cm.gray_r)
  plt.show()
  
  return  