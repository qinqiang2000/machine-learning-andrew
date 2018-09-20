import numpy as np

def check(f, x):
  print(f(x))

def func(x, a, b):
  return (x, a, b)

f = lambda W: func(W, 2, 3) #f(w) = loss

check(f, 9)