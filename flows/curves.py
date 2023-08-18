import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from flows.metrics import edge_lengths, dot_product

def line(t):
  return (t,t)

def circle(t):
  return (np.cos(t), np.sin(t))

def shifted_circle(t, v):
  return (np.cos(t) + v[0], np.sin(t) + v[1])

def level_circle_x(t, x):
  """
  Return a level unit circle at x in R^3
  """
  return (jnp.full(shape = len(t), fill_value=x), jnp.sqrt(1-x**2)*jnp.cos(t), jnp.sqrt(1-x**2)*jnp.sin(t))

def line_func(s, e):
  def out(t):
    return (s[0]+(e[0]-s[0])*t,s[1]+(e[1]-s[1])*t)
  return out

def scale_function(func, s):
  def out(t):
    o = func(t)
    return (o[0]*s, o[1]*s)
  return out

def discretize_curve(curve, n, range, closed_curve=True):
  """
  Return : An array of [xval, yval]
  Inputs:
      curve : function discretized by parameter t
      n : number of samples
      range : a tuple (start,stop)
  """
  if closed_curve:
    t = np.linspace(range[0], range[-1]-(range[-1]-range[0])/n, n)
    points = curve(t)

    return  np.transpose(points)
  else:
    t = np.linspace(range[0], range[-1], n)
    points = curve(t)
    return np.transpose(points)

def discretize_curve2(curve, n, rnge, closed_curve=True, metric=dot_product):
  """
  DISCLAIMER: i have seen odd behaviour from this function when you feed a non closed curve with closed_curve=True
  when in doubt, just use discretize_curve

  Return : An array of [xval, yval], evenly spaced as well.
  Inputs:
      curve : function discretized by parameter t
      n : number of samples
      rnge : a tuple (start,stop)
  """
  if closed_curve:
    t = np.linspace(rnge[0], rnge[-1]-(rnge[-1]-rnge[0])/n, n)
    points = np.transpose(np.array(curve(t)))
    L = edge_lengths(points, metric)

    accumulator = 0
    accumulated_L = []
    for i in L:
      accumulated_L.append(accumulator)
      accumulator += i
    accumulated_L.append(accumulator)

    increment = accumulator / n
    curr_index = 0
    curr_float = 0
    desired = 0
    parameter = 0
    OUT = []
    for i in range(n):
      temp_index = curr_index
      while accumulated_L[temp_index] <= desired:
        temp_index += 1
      n_float = ( desired - accumulated_L[temp_index - 1] ) / ( accumulated_L[temp_index] - accumulated_L[temp_index - 1] )
      parameter_increment = n_float + (temp_index - curr_index - 1) - curr_float
      parameter += parameter_increment
      OUT.append(parameter)

      curr_index = temp_index - 1
      curr_float = n_float
      desired += increment

    T = rnge[0] + (np.array(OUT)/n)*(rnge[1]-rnge[0])
    points = np.transpose(curve(T))


    return points
  else:
    t = np.linspace(range[0], range[-1], n)
    points = curve(t)
    return np.transpose(points)
