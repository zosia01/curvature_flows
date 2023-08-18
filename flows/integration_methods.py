def midpoint_rule(fun, interval, start_point, end_point):
  """
  Approximate the definite integral of a function (fun) evaluated
  from start_point to end_point using the midpoint rule.

  """
  midpoint = 0.5*(start_point + end_point)
  midpoint_approx = interval*fun(midpoint)
  return midpoint_approx

def trapezoidal_rule(fun, interval, start_point, end_point):
  """
  Approximate the definite integral of a function (fun) evaluated
  from start_point to end_point using the trapezoidal rule.

  """
  trapezoidal_approx = interval*0.5*(fun(start_point) + fun(end_point))
  return trapezoidal_approx

def simpsons_rule(fun, interval, start_point, end_point):
  """
  Approximate the definite integral of a function (fun) evaluated
  from start_point to end_point using Simpson's rule.
  """

  midpoint = 0.5*(start_point + end_point)
  simpsons_approx = (interval/6.0)*(fun(start_point) + 4*fun(midpoint) + fun(end_point))
  return simpsons_approx