import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
from flows.helper_functions import get_correct_gradient_function, plot_curve
from metrics import length_elements
from integration_methods import midpoint_rule
import numpy as np

class Atlas:
  '''
    A class to keep track of data from multiple charts.
    Holds a set of charts and transition functions between them.
  '''    
  def __init__(self, charts, transition_functions):
    '''
      Inputs:
        charts -> an array of Chart objects
        transition_functions -> a dictionary whose keys are tuples (i,j), where 0 <= i,j < len(charts), 
          and whose values are functions from R2 -> R2. transition_functions[(i,j)] is the function that
          represents the transition map from charts[i] to charts[j]
    '''
    self.chart = charts
    self.transition_function = transition_functions
    self.vmaped_transition_functions = {}
    self.tangent_maps = {}
    self.vmaped_tangent_maps = {}
    self.chart_overlaps = {}
    self.N = -1

    for K in self.transition_function.keys():
      self.vmaped_transition_functions[K] = vmap(lambda x: self.transition_function[K](x))
      self.tangent_maps[K] = (jit(grad(lambda x: self.transition_function[K](x)[0])),jit(grad(lambda x: self.transition_function[K](x)[1])))
      self.vmaped_tangent_maps[K] = vmap(lambda x: jnp.array([self.tangent_maps[K][0](x), self.tangent_maps[K][1](x)]))
      self.chart_overlaps[K] = (lambda x: self.chart[K[1]].indicator_function( self.vmaped_transition_functions[K](x) ))

  def curve_length(self):
    '''
      Returns the curve length, as calculated using data from all charts, weighed using the partition of unity
    '''
    total = 0
    for chart in self.chart:
        A = length_elements(chart.curve, metric=chart.metric)
        W = chart.active_guys * chart.PoUF(chart.curve)
        total += sum(A*(W))
    return total

  def get_dt_estimate(self):
    '''
      Returns an estimate for dt based on the highest edge length of the curve in all possible charts.
      Did not prove better than diffrax however.
    '''
    out = 0
    for chart in self.chart:
      A = length_elements(chart.curve, metric=chart.metric)
      M = chart.active_guys
      mask = M * jnp.roll(M, -1, axis=0) * jnp.roll(M, 1, axis=0)
      out = max(out, jnp.max(A * mask))
    return out**2

  def ready_all_charts(self):
    '''
      Makes sure all charts have a curve in them. Should run this command before
      doing any curve evolution stuff
    '''
    number_of_charts = len(self.chart)
    active_charts = []
    for i in range(number_of_charts):
      if self.chart[i].has_curve:
        active_charts.append(1)
      else:
        active_charts.append(0)

    if sum(active_charts) == 0:
      print('add a curve to one of the charts')
      return

    N = len(self.chart[active_charts.index(1)].curve)
    for i in range(number_of_charts):
      if active_charts[i] == 0:
        self.chart[i].set_points(np.random.rand(N,2))
        self.chart[i].active_guys = self.chart[i].active_guys*0
    self.N = N

  def add_to_chart(self, chart_number, curve):
    """
    sets chart[chart_number]'s curve to be equal to `curve`
    in the atlas
    """
    self.chart[chart_number].set_points(curve)
    self.N = self.chart[chart_number].N

  def align_charts(self):
    """
      Aligns curves in all charts so that they agree in their overlaps
      by averaging their values in the overlapping region
    """
    for repetition in range(4):
      temporary_positions = [] # we need to have a place to store the changes to the positions of each chart
      new_active_guys = [] # we need to also modify what points are active in each chart, since new points could enter a chart

      quotient = jnp.zeros((self.N))

      for i in range(len(self.chart)):
        self.chart[i].filter_points()
        temporary_positions.append(jnp.zeros((self.N, 2)))
        new_active_guys.append(jnp.zeros(self.N))
        quotient += self.chart[i].active_guys

      # for each transition function, in each overlapping region, average the positions of points according to the value of the partition of unity
      for K in self.transition_function.keys():
        domain = self.chart[K[0]]
        active = domain.active_guys*self.chart_overlaps[K](domain.curve) # makes sure to only look at active guys in the overlapping area

        new_active_guys[K[1]] = jnp.where(new_active_guys[K[1]] > active, new_active_guys[K[1]], active) # updates the active guys of the codomain chart
        new_curve = jnp.where(active>0.5, self.vmaped_transition_functions[K](domain.curve).transpose(), jnp.zeros((2, self.N))).transpose() # sets points
        # that arent in the domain of the transition map to zero, since they may be unpredictable
        # (for example with the stereographic projection transition map the point (0,0) will become 'Nan' and we dont want that in our curve points)

        weighing = active/quotient #domain.PoUF(domain.curve) # weighs the points
        #print(weighing)
        #print('but this is the other one')
        #print(active*domain.PoUF(domain.curve))
        temporary_positions[K[1]] += jnp.einsum('ij,i->ij', new_curve, weighing)

      for i in range(len(self.chart)):
        domain = self.chart[i]
        self.chart[i].curve = temporary_positions[i] + jnp.einsum('ij,i->ij', domain.curve, domain.active_guys/quotient)#domain.active_guys*domain.PoUF(domain.curve)) # weighted sum
        # update active guys
        domain.active_guys = jnp.where(new_active_guys[i] > domain.active_guys, new_active_guys[i], domain.active_guys)

    for i in range(len(self.chart)):
      self.chart[i].filter_points()

  def calculate_curve_velocities(self):
    """
    calculates curve velocitites in every chart
    """
    # we need arrays to keep track of the changes we want to do to to the velocities of each chart
    temporary_velocity = []
    for i in range(len(self.chart)):
      self.chart[i].calculate_curve_velocity()
      bang = jnp.zeros((self.N, 2))
      temporary_velocity.append(bang)

    # for each transition function U->V, we calculate the curve velocities is U and then use the pushforward to
    # see what the respective velocity would be in V
    for K in self.transition_function.keys():
      domain_chart = self.chart[K[0]] # this is U
      transformed_velocities = vmap(lambda m, v: jnp.matmul(m,v))(self.vmaped_tangent_maps[K](domain_chart.curve), domain_chart.curve_velocity) # the velocities are pushed forward

      # makes sure only velocities that are in the overlap of the chart are updated
      indicator = self.chart_overlaps[K](domain_chart.curve)
      MM = jnp.vstack((indicator,indicator)).transpose()
      temporary_velocity[K[1]] += jnp.where(MM < 1., jnp.zeros((self.N, 2)), transformed_velocities)

    # adjusts all velocities
    for i in range(len(self.chart)):
      self.chart[i].curve_velocity += temporary_velocity[i]


  def take_a_step(self, dt=1):
    '''
      Does all necessary things to evolve the curve in all charts of the atlas using timestep dt
    '''
    self.calculate_curve_velocities()

    for chart in self.chart:
      chart.curve = chart.curve - (chart.curve_velocity * dt)
      chart.filter_points()

    self.align_charts()

  def __str__(self):
    out = "So this the atlas... get ready...\nthese the chartskis\n\n"
    n = 0
    for c in self.chart:
      out += 'chart numero '
      out += str(n)
      out += ':\n'
      out += str(c)
      out += '\n\n'
      n += 1
    return out

class Chart:
  '''
    A class that holds metric data, partition of unity function data, boundary data,
    and coordinate data of a curve.
  '''
  curve = None
  curve_velocity = None
  active_guys = None
  has_curve = False
  PoUE = None
  N = -1

  def __init__(self, metric, indicator_function=(lambda x: 1), partition_of_unity=(lambda x: 1), integration_method=midpoint_rule):
    """
      Inputs:
        metric <= metric to be used, compatible with get_correct_gradient_function
        indicator_function <= function from Chart^n -> {0,1}^n telling you if you are inside the chart.
        partition_of_unity <= function from Chart^n -> [0,1]^n
    """
    self.metric = metric
    self.indicator_function = indicator_function
    self.G = (get_correct_gradient_function(metric))
    self.PoUF = partition_of_unity

  def set_points(self, C):
    '''
      Sets the chart's curve coordinates according to C.
      Inputs:
        C -> a (N,2) numpy (or jnp) array. Each row is a vertex of the curve
    '''
    self.curve = C
    self.active_guys = jnp.ones(len(C))
    self.filter_points()
    self.has_curve = True
    self.N = len(C)

  def calculate_curve_velocity(self):
    '''
      Calculates the chart's curves veloctiy according to the gradient of length,
      weighed by the partition of unity
    '''
    weighed_velocities = jnp.einsum('ij,i->ij', self.G(self.curve), self.PoUF(self.curve)).transpose()
    self.curve_velocity = jnp.where(self.active_guys>0.5, weighed_velocities, jnp.zeros((2, self.N))).transpose()

  def filter_points(self):
    '''
      Determines what points have left the chart, updates the chart's active guys
    '''
    stays = self.indicator_function(self.curve)
    self.active_guys = stays*self.active_guys
    self.PoUE = self.PoUF(self.curve)*self.active_guys

  def debug_view(self, measurement='bruh', G='bruh', scale=0.1, plotter=plt):
    '''
      Plots the chart's curve, along with its velocity if it has one, and some partition
      of unity information
    '''
    C = self.curve

    if measurement == 'bruh':
      measurement = self.active_guys*self.PoUF(self.curve)
    if G == 'bruh':
      G = self.curve_velocity

    plot_curve(C, plotter=plotter)
    color_jawn = jnp.vstack((measurement, 1. - measurement, measurement))
    plotter.scatter(C[:,0], C[:,1], s=30,c=color_jawn.transpose(), vmax=1,vmin=0)
    n = len(C)
    for i in range(n):
      plotter.annotate(str(i), (C[i,0], C[i,1]))

    if G != None:
      for i in range(n):
        plotter.plot([C[i,0],C[i,0]+G[i,0]*scale], [C[i,1],C[i,1]+G[i,1]*scale], color=(0.9,0.3,0.))
  def __str__(self):
    return f"Metric: {self.metric}\nCurve: \n{self.curve}\nWeights: \n{self.PoUF(self.curve)}\nActive Guys:\n{self.active_guys}"