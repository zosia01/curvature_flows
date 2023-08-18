import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from flows.metrics import edge_lengths,length_elements, length_func
from flows.integration_methods import midpoint_rule
from jax import grad, jit, vmap

def stereo_point_north(point):
  """
  Returns the stereographic projection (X,Y) from the
  North pole of a point (x,y,z)
  """
  x = point[0]
  y = point[1]
  z = point[2]
  return (x/(1-z), y/(1-z))

def stereo_point_south(point):
  """
  Returns the stereographic projection (X,Y) from the
  South pole of a point (x,y,z)
  """
  x = point[0]
  y = point[1]
  z = point[2]
  return (x/(1+z), y/(1+z))

def stereo_curve(curve):
  """
  Return the stereographic projection from the North Pole
  of each point (x,y,z) on a curve.
  """
  x_arr =  curve[:,0]
  y_arr =  curve[:,1]
  z_arr = curve[:,2]
  return jnp.transpose(jnp.array([jnp.divide(x_arr,jnp.ones(len(z_arr))-z_arr), jnp.divide(y_arr,jnp.ones(len(z_arr))-z_arr)]))

def inversion_curve(curve):
  """
  Returns the inverse of each point (x,y) on a plane curve relative to the unit circle.
  """
  x_arr =  curve[:,0]
  y_arr =  curve[:,1]
  return jnp.transpose(jnp.array([jnp.divide(x_arr, jnp.square(x_arr) + jnp.square(y_arr)), jnp.divide(y_arr, jnp.square(x_arr) + jnp.square(y_arr))]))

def stereo_point_inverse(point):
  x = point[0]
  y = point[1]
  m = 1+x**2+y**2
  return (2*x/m,2*y/m,(-1+x**2+y**2)/m)

def stereo_curve_inverse(curve):
  x_arr =  curve[:,0]
  y_arr =  curve[:,1]
  m = 1+x_arr**2+y_arr**2
  return jnp.transpose(jnp.array([2*jnp.divide(x_arr,m),2*jnp.divide(y_arr,m),jnp.divide((-1+x_arr**2+y_arr**2),m)]))


#3-Sphere stereographic projection

def hs_stereo_point(point):
  """
  Returns the stereographic projection (X,Y,Z) from the
  point (1,0,0,0) of a point (x,y,z,w)
  """
  x = point[0]
  y = point[1]
  z = point[2]
  w = point[3]
  return (x/(1-w), y/(1-w), z/(1-w))

def hs_stereo_curve(curve):
  """
  Returns the stereographic projection (X,Y,Z) from the
  point (1,0,0,0) of each point (x,y,z,w) on a curve.
  """
  x_arr =  curve[:,0]
  y_arr =  curve[:,1]
  z_arr = curve[:,2]
  w_arr = curve[:,3]
  return jnp.transpose(jnp.array([jnp.divide(x_arr,jnp.ones(len(w_arr))-w_arr), jnp.divide(y_arr,jnp.ones(len(w_arr))-w_arr), jnp.divide(z_arr,jnp.ones(len(w_arr))-w_arr)]))

def hs_random_curve(t):
  return (jnp.full(shape=len(t), fill_value=0.5), 0.5*jnp.cos(t), 0.5*jnp.sin(t), jnp.full(shape=len(t), fill_value=0.5))

