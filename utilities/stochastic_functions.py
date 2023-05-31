import numpy as np
from scipy.stats import truncnorm

from utilities.constants import MOPTA_CONSTANTS


def ev_charging_probabilities(ranges: np.ndarray) -> np.ndarray:
    """
    Returns the probability of an ev with a certain range to go to a charging station
    :param ev_range: array of ev ranges
    :return: the probability to go to a charger
    """
    charge_prob = np.exp(-MOPTA_CONSTANTS['lambda_charge'] ** 2 * (ranges - 20) ** 2)
    return charge_prob


def ev_charging(ranges: np.ndarray, charging_probabilites: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Determines whether a car wants to charge based on its range and a random number.
    :param ev_range: ranges of cars
    :param seed: use seed for ranges
    :return: boolean array if car goes charging or not
    """
    if seed:
        np.random.seed(seed)
    return np.random.uniform(size=len(ranges)) <= charging_probabilites


def generate_ranges(num: int, seed: int = None) -> np.ndarray:
    """
    Generate num ranges from truncated normal distribution.
    :param seed: use seed for ranges
    :param num: number of ranges need
    :return: ranges from truncated normal distribution
    """
    mu = MOPTA_CONSTANTS['mu_range']  # mean
    sigma = MOPTA_CONSTANTS['sigma_range']  # standard deviation
    lb = MOPTA_CONSTANTS['lb_range']  # lower bound
    ub = MOPTA_CONSTANTS['ub_range']  # upper bound
    if seed:
        np.random.seed(seed)
    t_norm = truncnorm((lb - mu) / sigma, (ub - mu) / sigma, loc=mu, scale=sigma)
    return t_norm.rvs(num)

