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
  return jnp.where(x>0., jnp.exp(-1/x), 0.)

def smoothstep(x, r1=0, r2=1):
  y = (x-r1)/(r2-r1)
  f = 6*(y**5)-15*(y**4)+10*(y**3)#3*(y**2)-2*(y**3)
  return jnp.where(y<1,jnp.where(y<0, 0, f), 1)

def plot_curve_and_velocities(curve, velocities, velocities2=None, scale=1., plotter=plt, curve_color=(0.,0.,0.), velocities_color=(1.,0.,0.), velocities2_color=(0.,1.,0.)):
  '''
    plots the curve given by `curve` (a (N,2) jnp.array) and velocities (another (N,2) jnp.array)
  '''
  if plotter == plt:
    plotter.gca().set_aspect('equal')
  
  plotter.plot(curve[:,0], curve[:,1], color=curve_color)
  plotter.plot([curve[-1,0],curve[0,0]],[curve[-1,1],curve[0,1]], color=curve_color)
  for i in range(len(velocities)):
    plotter.plot([curve[i,0],curve[i,0]+scale*velocities[i,0]],[curve[i,1],curve[i,1]+scale*velocities[i,1]], color=velocities_color)
  
  if not(velocities2 == None):
    for i in range(len(velocities)):
      plotter.plot([curve[i,0],curve[i,0]+scale*velocities2[i,0]],[curve[i,1],curve[i,1]+scale*velocities2[i,1]], color=velocities2_color)