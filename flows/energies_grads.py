import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from flows.metrics import edge_lengths,length_elements, length_func
from flows.integration_methods import midpoint_rule
from jax import grad, jit, vmap

# Functions to correct gradients
def get_inverse_coefficients(p, metric):
  return jnp.linalg.inv(metric(p))

def correct_gradient(G, C, metric, integration_method=midpoint_rule):
  L = 1 / length_elements(C, metric=metric, integration_method=integration_method)
  M = vmap(lambda x: get_inverse_coefficients(x, metric))(C)
  out = vmap(lambda x, m: jnp.matmul(x, m))(G, M)
  return jnp.einsum('ij,i->ij', out, L)

# total curvature energy function for curve-straightening flow
def curvature_func(curve, metric, integration_method):
  """
  Returns an approximation of the elastic energy (for curve-straightening flow).
  """
  grad_len = jit(grad(lambda curve: length_func(curve, metric, integration_method)))
  G = jit(lambda x: correct_gradient(grad_len(x), x, metric))
  grads = G(curve)
  ki_arr = vmap(lambda L_grad, p: jnp.matmul(L_grad, jnp.matmul(metric(p), L_grad)))(grads, curve)
  return (1/2)*jnp.sum(jnp.multiply(length_elements(curve, metric),ki_arr))
