import matplotlib.pyplot as plt

  # Plot training data
def plotData(X, y):
  plt.plot(X, y, 'rx')
  plt.xlabel('Change in water level (x)')
  plt.ylabel('Water flowing out of the dam (y)')