import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp

def phi(point):
  """
  Returns the polar angle (in spherical coordinates)
  of a point (x,y,z) on a sphere.
  """
  return jnp.arccos(point[2]/jnp.sqrt(point[0]**2+ point[1]**2 + point[2]**2))

def phi_rotated(point):
  """
  Returns the polar angle (in spherical coordinates)
  of a point (x,y,z) on a sphere, with respect to negative x-axis.
  """
  return jnp.arccos(-point[0]/jnp.sqrt(point[0]**2+ point[1]**2 + point[2]**2))

def phi_rotated_curve(curve):
  curve = curve[:]
  return jnp.arccos(-curve[:,0]/jnp.sqrt(curve[:,0]**2+ curve[:,1]**2 + curve[:,2]**2))

def phi_curve(curve):
  curve = curve[:]
  return jnp.arccos(curve[:,2]/jnp.sqrt(curve[:,0]**2+ curve[:,1]**2 + curve[:,2]**2))

def mean_phi_curve(curve):
  curve = curve[:]
  return jnp.arccos(jnp.mean(curve[:,2]/jnp.sqrt(curve[:,0]**2+ curve[:,1]**2 + curve[:,2]**2)))

def mean_phi_plane_curve(curve):
  """
  Returns polar angle in spherical coordinates of
  curve (x_i, y_i) in chart of stereographic projection
  """
  r = np.mean(np.sqrt(curve[:,0]**2 + curve[:,1]**2))
  return jnp.arcsin(2*r/(1+r**2))