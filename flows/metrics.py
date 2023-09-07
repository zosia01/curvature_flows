import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from jax import grad, jit, vmap
from flows.integration_methods import midpoint_rule


def stereographic_metric(p):
  """
  Calculate the induced Riemannian metric of vec1 and vec2
  at point p=(x,y) on the stereographic projection of the
  sphere from the North Pole.
  """
  x = p[0]
  y = p[1]
  m = 4/((1+x**2+y**2)**2)
  return m*np.identity(2)

def mercator_metric(p):
  """
  Calculate the induced Riemannian metric of vec1 and vec2
  at point p=(x,y) on the mercator projection of the
  sphere.
  """
  return np.array([1,0], [0, (jnp.sin(p[0])**2)])

def get_torus_metric(r,R):
  """
    Returns torus metric function for the given values of r and R
  """
  def out(p):
    return jnp.array([[(R+r*jnp.cos(p[1]))**2,0.],[0.,r*r]])
  return out

def dot_product(p):
  """
  Apply the dot product metric tensor at a point (p) on two vectors (v and w).

  """
  return np.identity(2)

def hs_stereo_metric(p):
  m = 4/((1+jnp.dot(p, p))**2)
  return m*np.identity(3)

def angenent_metric(p):
  """
  Apply the Angenent torus metric in the upper half plane
  """
  r = p[0]
  z = p[1]
  return (1/4)*(r**2*jnp.exp(-(r**2+z**2)/2))*jnp.identity(2)



def length_func(curve, metric, integration_method):
  """
  Return: Length of the discretized curve curve, as described in notes.
  Input:
    curve: an numpy array of vertices, representing a discretized curve
    metric: a symmetric 2-tensor field, or a function (v,w,p)-> r where r
    is a real number, v,w are vectors in TpM, and p is a point in our manifold M
  """

  shifted_curve = jnp.roll(curve, -1, axis = 0)
  edges = curve - shifted_curve
  lengths = vmap(lambda edge, start_point, end_point: integration_method(lambda p: jnp.sqrt(jnp.matmul(edge, jnp.matmul(metric(p), edge))), 1, start_point, end_point))(edges, curve, shifted_curve)
  total_length = jnp.sum(lengths)

  return total_length

def length_elements(curve, metric=dot_product, integration_method=midpoint_rule):
  L = edge_lengths(curve, metric, integration_method)
  return (L + jnp.roll(L, 1, axis=0))*0.5

def edge_lengths(curve, metric, integration_method=midpoint_rule):
  shifted_curve = jnp.concatenate((curve[1:],curve[:1]), axis=0)
  edges = shifted_curve - curve
  s = vmap(lambda e, p: jnp.matmul(e, jnp.matmul(metric(p), e)))
  O = integration_method(lambda x: jnp.sqrt(s(edges, x)), 1, curve, shifted_curve)
  return O

def get_inverse_coefficients(p, metric):
  return jnp.linalg.inv(metric(p))

def correct_gradient(G, C, metric, integration_method=midpoint_rule):
  L = 1 / length_elements(C, metric=metric, integration_method=integration_method)
  M = vmap(lambda x: get_inverse_coefficients(x, metric))(C)
  out = vmap(lambda x, m: jnp.matmul(x, m))(G, M)
  return jnp.einsum('ij,i->ij', out, L)

