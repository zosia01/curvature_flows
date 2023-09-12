import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
from flows.helper_functions import get_correct_gradient_function, plot_curve
from flows.metrics import length_elements
from flows.integration_methods import midpoint_rule, trapezoidal_rule
import numpy as np

class Chart:
  def __init__(self, metric, partition_of_unity, support=(lambda x: True)):
    """
      partition of unity <= a function f: R2 -> [0,1]
      support <= a function f: R2 -> {True, False}
    """
    self.metric = metric
    self.partition_of_unity = partition_of_unity
    self.indicator_function = support

class Atlas:
  def __init__(self, charts, transition_functions):
    '''
    Takes in a list of chart objects and a dictionary of transition functions.
    You only need to include transition functions in between charts that actually overlap,
    and you dont need to include the transition function to the chart to itself (the identity transition function).
    In fact, doing so will cause an error.
    '''
    self.charts = charts
    self.transition_functions = {}
    self.tangent_map = {}
    self.vmaped_tangent_maps = {}
    self.curve_coordinates = None
    self.curve_velocities = None
    self.active_guys = None
    self.N = -1
    self.number_of_charts = len(charts)
    self.G = None

    for K in transition_functions.keys():
      self.transition_functions[K] = jit(vmap(transition_functions[K]))
      self.tangent_map[K] = (jit(grad(lambda x: transition_functions[K](x)[0])),jit(grad(lambda x: transition_functions[K](x)[1])))
      self.vmaped_tangent_maps[K] = jit(vmap( lambda x: jnp.array( [self.tangent_map[K][0](x), self.tangent_map[K][1](x)] ) ))

    # TODO: somehow check that overlaps and partitions of unity make sense

  def get_indicator_function(self):
    return self.get_indicator_function_pure(self.curve_coordinates)

  def get_indicator_function_pure(self, C):
    out = []
    for i in range(self.number_of_charts):
      out.append(
          vmap(lambda x: self.charts[i].indicator_function( x ))(C[i])
      )
    return jnp.array(out)

  def pure_weights(self, BLOCK, ACTIVE):
    out = []
    for i in range(self.number_of_charts):
      out.append(
                  vmap(lambda x: self.charts[i].partition_of_unity(x))(BLOCK[i]) *
                  jnp.where( ACTIVE[i] , 1., 0.)
                )
    return jnp.array(out)

  def weights(self, BLOCK):
    out = []
    for i in range(self.number_of_charts):
      out.append(
                  vmap(lambda x: self.charts[i].partition_of_unity(x))(BLOCK[i]) *
                  jnp.where( self.active_guys[i] , 1., 0.)
                )
    return jnp.array(out)

  def pure_weighted_length_elements(self, BLOCK, ACTIVE):
    out = []
    for i in range(self.number_of_charts):
        out.append(length_elements(BLOCK[i], self.charts[i].metric, trapezoidal_rule))
    return jnp.array(out) * self.pure_weights(BLOCK, ACTIVE)

  def weighted_length_elements(self, BLOCK):
    out = []
    for i in range(self.number_of_charts):
        out.append(length_elements(BLOCK[i], self.charts[i].metric, trapezoidal_rule))# *
    return jnp.array(out) * self.weights(BLOCK)

  def L(self, BLOCK):
    return jnp.sum(self.weighted_length_elements(BLOCK))

  def pure_L(self, BLOCK, ACTIVE):
    return jnp.sum(self.pure_weighted_length_elements(BLOCK, ACTIVE))

  def pure_take_a_step(self, BLOCK, ACTIVE, dt):
    V = self.calculate_velocities_pure(BLOCK, ACTIVE)
    new_C = BLOCK + dt*V
    return self.align_charts_pure(new_C, ACTIVE), V

  def take_a_step(self, dt):
    CA, V = self.jitted_take_a_step(self.curve_coordinates, self.active_guys, dt)
    self.curve_coordinates = CA[0]
    self.curve_velocities = V
    self.active_guys = CA[1]

  def align_charts_pure(self, C, A):
    for repeats in range(3):
      A = self.filter_points_pure(C,A)

      # make container for new coordinates
      new_curve_coordinates = []
      new_active_guys = []
      for chart in range(self.number_of_charts):
        new_curve_coordinates.append(jnp.zeros((self.N, 2)))
        new_active_guys.append(jnp.zeros(self.N)==1.)
      quotient = jnp.sum(jnp.where(A, 1., 0.), 0)
      weighing = 1. / quotient

      # for all transition functions, find corresponding coordinates.
      for K in self.transition_functions.keys():
        i = K[0]
        j = K[1]

        active = jnp.logical_and(A[i], vmap(lambda p: self.charts[j].indicator_function(p))(self.transition_functions[K](C[i])))
        new_active_guys[j] |= active

        new_curve = jnp.where(active, self.transition_functions[K](C[i]).transpose(), jnp.zeros((2,self.N))).transpose()
        new_curve_coordinates[j] += jnp.einsum('ij,i->ij', new_curve, weighing)

      new_curve_coordinates = jnp.array(new_curve_coordinates)
      new_curve_coordinates += vmap(lambda curve_in_chart: jnp.einsum('ij,i->ij', curve_in_chart, weighing))(C)
      A = jnp.logical_or(A, jnp.array(new_active_guys))
    A = self.filter_points_pure(C, A)
    return C, A

  def align_charts(self):
    C, A = self.align_charts_pure(self.curve_coordinates, self.active_guys)
    self.curve_coordinates = C
    self.active_guys = A

  def filter_points_pure(self, C, A):
    return jnp.logical_and(self.get_indicator_function_pure(C), A)

  def filter_points(self):
    self.active_guys = self.filter_points_pure(self.curve_coordinates, self.active_guys)

  def DL_of(self, DL, V):
    out = []
    for i in range(self.number_of_charts):
      row = vmap(lambda d,v: jnp.dot(d,v))(DL[i],V[i])
      out.append(row)
    return jnp.array(out)

  def L2_inner_product(self, V1, V2):
    out = []
    W = self.weighted_length_elements(self.curve_coordinates)
    for i in range(self.number_of_charts):
      row = vmap(lambda v1,v2,p: jnp.matmul(v1,jnp.matmul(self.charts[i].metric(p), v2)))(V1[i],V2[i],self.curve_coordinates[i])
      out.append(row)
    return sum(sum(jnp.array(out)*W))

  def calculate_velocities_pure(self, C, A):
    DL = -self.pure_G(C, A)

    E_x, E_y = self.pure_make_consistent_vector_field(C,A)

    PoU = self.pure_weights(C, A)
    C_i = jnp.sum(self.pure_weighted_length_elements(C, A),0)

    p = jnp.sum(self.DL_of(DL, E_x),0)
    q = jnp.sum(self.DL_of(DL, E_y),0)

    coefficients = []
    for alpha in range(self.number_of_charts):
      coefficients.append(
          vmap(lambda c, p_i, q_i, e_xi, e_yi: jnp.matmul(jnp.linalg.inv(jnp.matmul(jnp.vstack((e_xi, e_yi)), jnp.matmul(self.charts[alpha].metric(c), jnp.vstack((e_xi, e_yi))))), jnp.array([p_i,q_i])))(C[alpha], p ,q, E_x[alpha], E_y[alpha])
      )
    coefficients = jnp.sum(vmap(lambda c, PoU_a: jnp.einsum('ij,i->ij',c,PoU_a))(jnp.array(coefficients), PoU), 0)
    coefficients = jnp.einsum('ij,i->ij', coefficients, 1/C_i)

    V = vmap(lambda ex, ey: jnp.einsum('ij,i->ij', ex, coefficients[:,0]) + jnp.einsum('ij,i->ij', ey, coefficients[:,1]))(E_x, E_y)
    return V

  def pure_make_consistent_vector_field(self, C, A):
    used_bruhs = jnp.zeros(self.N)
    e_x = jnp.vstack((jnp.ones(self.N),jnp.zeros(self.N))).transpose()
    e_y = jnp.vstack((jnp.zeros(self.N),jnp.ones(self.N))).transpose()

    out_x = []
    out_y = []

    for i in range(self.number_of_charts):
      out_x.append(jnp.zeros((self.N,2)))
      out_y.append(jnp.zeros((self.N,2)))

    for i in range(self.number_of_charts):
      curve = C[i]
      M = A[i]

      added_x = jnp.where(jnp.logical_and(used_bruhs<0.5, M), e_x.transpose(), 0.).transpose()
      added_y = jnp.where(jnp.logical_and(used_bruhs<0.5, M), e_y.transpose(), 0.).transpose()
      out_x[i]+= added_x
      out_y[i]+= added_y

      used_bruhs = jnp.where(M, 1., used_bruhs)

      for K in self.transition_functions.keys():
        if K[0]==i:
          matrices = self.vmaped_tangent_maps[K]( curve )
          
          # the velocities are pushed forward
          transformed_velocities_x = vmap(lambda m, v: jnp.matmul(m,v))(matrices, added_x) 
          transformed_velocities_y = vmap(lambda m, v: jnp.matmul(m,v))(matrices, added_y)

          # makes sure only velocities that are in the overlap of the chart are updated
          out_x[K[1]] += jnp.where(M, transformed_velocities_x.transpose(), 0.).transpose()
          out_y[K[1]] += jnp.where(M, transformed_velocities_y.transpose(), 0.).transpose()

    return jnp.array(out_x), jnp.array(out_y)


  def calculate_velocities(self):
    self.curve_velocities = self.calculate_velocities_pure(self.curve_coordinates, self.active_guys)

  def make_consistent_vector_field(self):
    return self.pure_make_consistent_vector_field(self.curve_coordinates, self.active_guys)

  def set_curve(self, curve_coordinates, active_guys):
    self.curve_coordinates = jnp.array(curve_coordinates)
    self.N = self.curve_coordinates.shape[1]
    self.active_guys = jnp.logical_and( jnp.array(active_guys), self.get_indicator_function() )
    self.G = jit(grad(lambda x: self.L(x)))
    self.pure_G = jit(grad(self.pure_L, argnums=0))
    self.jitted_take_a_step = jit(self.pure_take_a_step)

  def print_active_guys(self, A=None):
    print('#  |',end='')
    for i in range(self.number_of_charts):
      print('c {} |'.format(i),end='')
    print('')
    for i in range(self.N):
      if i<10:
        print('{}  |'.format(i),end='')
      else:
        print('{} |'.format(i),end='')
      for j in range(self.number_of_charts):
        if A==None:
          if self.active_guys[j][i]:
            print('1',end='')
          else:
            print('0',end='')
        else:
          if A[j][i]:
            print('1',end='')
          else:
            print('0',end='')

        print('   |',end='')
      print('')
