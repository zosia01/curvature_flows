import matplotlib.pyplot as plt
from flows.energies_grads import correct_gradient
from flows.metrics import length_func
from flows.integration_methods import midpoint_rule
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

def smooth_f(x):
# Smooth function on the real line that is zero for t<=0 and positive for t>0
  #if x > 0.:
  #  return np.exp(-1/x)
  #else:
  #  return 0.
  return jnp.where(x>0., jnp.exp(-1/x), 0.)

def smoothstep(x, r1=0, r2=1):
  '''
  smoothstep(x, r1, r2) is a smooth function of x satisfying
  smoothstep(x, r1, r2) = 0 for x < r1
  0 < smoothstep(x, r1, r2) < 1 for r1 < x < r2
  smoothstep(x, r1, r2) = 1 for x > r2
  '''
  o = r1
  s = 1/(r2-r1)
  t = smooth_f(s*(x-o))
  return t/(t+smooth_f(1-(s*(x-o))))

