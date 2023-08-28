import matplotlib.pyplot as plt
from flows.energies_grads import correct_gradient
from flows.metrics import length_func
from integration_methods import midpoint_rule
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap

def plot_curve(curve, plotter=plt, closed_curve=True):
  """
    Helper function to not have ugly code
  """
  #plotter.gca().set_aspect('equal')
  if closed_curve:
    C = np.concatenate((curve, curve[:1]))
    plotter.plot(C[:,0], C[:,1])
  else:
    plotter.plot(curve[:,0], curve[:,1])

def get_correct_gradient_function(metric, integration_method=midpoint_rule):
  p = (grad(lambda x: length_func(x, metric, integration_method)))
  return jit(lambda x: correct_gradient(p(x), x, metric))