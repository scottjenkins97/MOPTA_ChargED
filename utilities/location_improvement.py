import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from utilities.helper_functions import get_distance_matrix, geometric_median

import warnings

# filter warning for scipy
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
def find_optimal_location(allocated_locations, allocated_ranges):
    """
    This function finds a better location for a given allocated location and range.
    This problem is convex and can be solved quickly.
    :param allocated_locations: locations allocated to a charger
    :param allocated_ranges: the ranges of the cars at these locations
    :return: best location given those ranges
    """

    # first check whether the geometric mean is feasible -> already minimises and no computation needed
    geom = geometric_median(allocated_locations)
    distances = get_distance_matrix(allocated_locations, np.array([geom])).flatten()
    if np.all(distances <= allocated_ranges):
        return geom

    # geometric mean is not feasible -> set up scipy problem
    objective = lambda x: get_distance_matrix(allocated_locations, np.array([x])).sum()
    constraint = NonlinearConstraint(lambda x: get_distance_matrix(allocated_locations, np.array([x])).flatten(), 0,
                                     allocated_ranges)
    # derivatives
    distance = lambda x: np.linalg.norm(allocated_locations - x, axis=1)
    part_x1 = lambda x: x[0] - allocated_locations[:, 0]
    part_x2 = lambda x: x[1] - allocated_locations[:, 1]
    # jacobian
    jac = lambda x: (np.sum(part_x1(x) / distance(x)), np.sum(part_x2(x) / distance(x)))
    # hessian
    upper_left = lambda x: np.sum(part_x2(x) ** 2 / distance(x) ** 3)
    upper_right = lambda x: -np.sum(part_x1(x) * part_x2(x) / distance(x) ** 3)
    lower_right = lambda x: np.sum(part_x1(x) ** 2 / distance(x) ** 3)
    hess = lambda x: np.matrix([[upper_left(x), upper_right(x)], [upper_right(x), lower_right(x)]])

    # solve problem using trust region method
    sol_scipy = minimize(objective, geom, constraints=constraint, method='trust-constr', jac=jac, hess=hess)
    if not np.all(get_distance_matrix(allocated_locations, np.array([sol_scipy.x])).flatten() <= allocated_ranges):
        raise ValueError('Optimal solution of subproblem is not feasible')

    return sol_scipy.x
