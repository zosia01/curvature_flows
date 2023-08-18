import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp


def mercator_point(point):
  """
  Returns the mercator projection (theta, phi) of a point (x, y, z) in the sphere
  """
  x = point[0]
  y = point[1]
  z = point[2]
  theta = jnp.arccos(z/np.sqrt(np.dot(point, point)))
  phi = jnp.arctan2(y, x)
  if phi < 0:
    phi = 2*jnp.pi - phi
  return (theta, phi)

def mercator_curve_inverse(curve):
  theta_arr =  curve[:,0]
  phi_arr = curve[:,1]
  inverse_curve = jnp.transpose(jnp.array([jnp.sin(theta_arr)*jnp.cos(phi_arr), jnp.sin(theta_arr)*jnp.sin(phi_arr), jnp.cos(theta_arr)]))
  return inverse_curve

def mercator_curve(curve):
  x_arr = curve[:, 0]
  y_arr = curve[:, 1]
  z_arr = curve[:, 2]
  theta_arr = jnp.arccos(jnp.divide(z_arr, jnp.sqrt(np.square(x_arr) + np.square(y_arr) + jnp.square(z_arr))))
  phi_arr = jnp.arctan2(y_arr, x_arr)
  for i, phi in enumerate(phi_arr):
    if phi < 0:
      phi_arr = phi_arr.at[i].set(2*jnp.pi + phi)
  return jnp.transpose(jnp.array([theta_arr, phi_arr]))