import logging
import time
from typing import Iterable, List, Tuple, Callable

import numpy as np
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from tqdm import tqdm

from utilities.constants import MOPTA_CONSTANTS
from utilities.helper_functions import get_distance_matrix, get_indice_sets_stations, compute_maximum_matching
from utilities.location_improvement import find_optimal_location
from utilities.stochastic_functions import ev_charging, ev_charging_probabilities, generate_ranges

# create logger
logger = logging.getLogger(__name__)
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(levelname)s: %(message)s')
# add formatter to console handler, ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class MOPTASolver:
    def __init__(self, car_locations: np.ndarray, loglevel=logging.INFO,
                 build_cost: float = MOPTA_CONSTANTS['build_cost'],
                 maintenance_cost: float = MOPTA_CONSTANTS['maintenance_cost'],
                 drive_cost: float = MOPTA_CONSTANTS['drive_cost'],
                 charge_cost: float = MOPTA_CONSTANTS['charge_cost'],
                 service_level: float = MOPTA_CONSTANTS['service_level'],
                 station_ub: int = MOPTA_CONSTANTS['station_ub'],
                 fixed_station_number: int = None,
                 streamlit_callback: Callable = None
                 ):
        """

        :param car_locations: the locations of the cars
        :param loglevel: logging level, e.g., logging.DEBUG, logging.INFO
        :param build_cost: the cost for building a location
        :param maintenance_cost: the cost for maintaining per charger at a location
        :param drive_cost: the cost per drive mile
        :param charge_cost: the cost per charged mile
        :param service_level: the percentage of cars that need to be charged
        :param station_ub: the number of cars, that can be charged at a location
        :param fixed_station_number: if wanted to specify the number of locations
        :param streamlit_callback: function to update streamlit user interface
        """
        logger.setLevel(level=loglevel)
        ch.setLevel(level=loglevel)

        # Sanity checks:
        # Car locations are in R^2 and there are at least two car locations
        if car_locations.shape[0] == 1:
            raise ValueError('Please add more than one car location.')
        if car_locations.shape[1] != 2:
            raise ValueError('Please add two dimensional car locations.')

        # Check whether service level is within (0,1]
        if service_level <= 0 or service_level > 1:
            raise ValueError('Service level should be within (0,1].')

        self.car_locations = car_locations
        self.n_cars = len(self.car_locations)

        # Tightest grid
        self.x_min = np.min(self.car_locations[:, 0])  # most left car
        self.x_max = np.max(self.car_locations[:, 0])  # most right car
        self.y_min = np.min(self.car_locations[:, 1])  # most down car
        self.y_max = np.max(self.car_locations[:, 1])  # most up car

        # generate all lists need for samples
        self.l = []  # number of cars in samples
        self.I = []  # indices of cars in samples
        self.R = []  # ranges of cars in samples
        self.D = []  # distance matrix of cars in samples
        self.reachable = []  # reachable matrix of cars in samples
        self.X = []  # car locations in samples
        self.S = None  # indices of samples
        self.n_samples = 0  # number of samples

        # charging locations
        self.L = None  # charging locations
        self.w = None  # number of charging locations
        self.J = None  # indices of charging locations

        # model
        self.m = None  # model
        self.b = None  # build binary variables
        self.n = None  # number integer variables
        self.u = None  # allocation binary variables
        self.station_ub = station_ub  # upper bound on number of stations
        self.build_cost_param = build_cost  # build cost
        self.maintenance_cost_param = maintenance_cost  # maintenance cost
        self.charge_cost_param = charge_cost  # charge cost
        self.drive_charge_cost_param = charge_cost + drive_cost  # drive + charge cost

        # objective terms
        self.build_cost = None
        self.maintenance_cost = None
        self.drive_charge_cost = None
        self.fixed_charge_cost = None

        # constraints
        self.fixed_station_number = fixed_station_number  # fixed number of stations
        self.service_level = service_level  # service level
        self.fixed_station_number_constraint = None  # fixed number of stations constraint
        self.b_n_constraint = None  # n only positive if also b positive
        self.n_b_constraint = None  # b only positive if also n positive
        self.max_queue_constraints = []  # max queue length constraints
        self.allocation_constraints = []  # allocation constraints (allocated to up to one charging station)
        self.service_constraints = []  # service constraints (at least XX% are serviced)

        # kpis
        self.kpi_build = None
        self.kpi_maintenance = None
        self.kpi_drive_charge = None
        self.kpi_avg_drive_distance = None
        self.kpi_fixed_charge = None
        self.kpi_total = None

        # solutions
        self.solutions = []
        self.objective_values = [np.inf]
        self.added_locations = []  # list of lists of the added locations per iteration

        # streamlit
        self.streamlit_callback = streamlit_callback  # callback function for streamlit

    def add_initial_locations(self, n_stations, mode='random', verbose=0, seed=None) -> None:
        """
        Add initial locations to the model
        :param n_stations: number of locations to add
        :param mode: random, k-means, k-means-constrained
        :param verbose: verbosity mode
        :param seed: seed for random state
        """

        if mode == 'random':
            logger.debug('Adding random locations.')
            # random generator
            rng = np.random.default_rng(seed=seed)
            # scale 2D random locations to grid
            new_locations = rng.random((n_stations, 2)) * np.array(
                [self.x_max - self.x_min, self.y_max - self.y_min]) + np.array([self.x_min, self.y_min])

        elif mode == 'k-means':
            logger.debug(f'Adding {n_stations} k-means locations.')
            kmeans = KMeans(
                n_clusters=n_stations,
                n_init=1,
                random_state=seed,
                verbose=verbose
            )
            new_locations = kmeans.fit(self.car_locations).cluster_centers_

        elif mode == 'k-means-constrained':
            logger.debug(f'Adding {n_stations} k-means-constrained locations.')
            kmeans_constrained = KMeansConstrained(
                n_clusters=n_stations,
                size_max=self.station_ub * 2 / MOPTA_CONSTANTS['mu_charging'],  # use expected number of charging cars
                n_init=1,
                random_state=seed,
                verbose=verbose
            )
            new_locations = kmeans_constrained.fit(self.car_locations).cluster_centers_

        else:
            raise Exception(
                'Invalid mode for initial locations. Choose between "random", "k-means" or "k-means-constrained".'
            )

        # add new locations to existing locations or create self.L
        if self.L is None:
            self.L = new_locations
        else:
            self.L = np.concatenate((self.L, new_locations))

        self.w = len(self.L)
        self.J = range(self.w)
        self.added_locations.append(self.L)

    def get_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a sample of ranges, charging prob and charging (bool) for all cars.
        :return: ranges, charging_probability and charging (bool)
        """
        ranges = generate_ranges(num=self.n_cars)
        charging_prob = ev_charging_probabilities(ranges=ranges)
        charging = ev_charging(ranges=ranges, charging_probabilites=charging_prob)
        return ranges, charging_prob, charging

    def add_sample(self):
        """
        Adds a sample of charging values to the problem, which is used to optimise over.
        """
        logger.debug('Adding sample.')

        if self.L is None:
            raise Exception('Please add initial locations before adding samples.')

        # get sample
        ranges, charging_prob, charging = self.get_sample()

        # mask cars that are actually charging
        charging_cars = self.car_locations[charging]
        self.l.append(len(charging_cars))
        self.I.append(range(len(charging_cars)))
        self.R.append(ranges[charging])
        self.D.append(get_distance_matrix(charging_cars, self.L))
        self.reachable.append((self.D[-1].T <= self.R[-1]).T)
        self.X.append(charging_cars)

        # Update sample sets and number of samples
        if self.S is None:
            self.S = range(1)
            self.n_samples = 1
        else:
            self.S = range(self.S.stop + 1)  # update sample set
            self.n_samples += 1

    def add_samples(self, num: int):
        for _ in range(num):
            self.add_sample()
        logger.info(f'Added {num} samples. Total number of samples: {self.n_samples}.')

    def create_b(self, K: Iterable):
        """
        Create binary variables b_k for each location k in K
        :param K: Iterable of (some) locations
        :return: b
        """
        created_b = np.array([self.m.binary_var(name=f'b_{k}') for k in K])
        if self.b is None:
            # if b didn't exist yet, return created b
            return created_b
        else:
            # if b already exists, append created b to existing b
            return np.append(self.b, created_b)

    def create_n(self, K: Iterable):
        """
        Create integer variables n_k for each location k in K
        :param K: Iterable of (some) locations
        :return: n
        """
        created_n = np.array([self.m.integer_var(name=f'n_{k}') for k in K])
        if self.n is None:
            return created_n
        else:
            return np.append(self.n, created_n)

    def create_u_s(self, s: int, K: Iterable):
        created_u_s = np.array(
            [self.m.binary_var(name=f'u_{s}_{i}_{k}') if self.reachable[s][i, k] else 0 for i in self.I[s] for k in K]
        )
        created_u_s = created_u_s.reshape(self.l[s], len(K))
        try:
            return np.concatenate((self.u[s], created_u_s), axis=1)
        except:
            return created_u_s

    def add_b_n_constraints(self, K: Iterable):
        logger.debug('Adding b - n constraints (n <= b * station upperbound).')
        return self.m.add_constraints((self.n[k] <= self.b[k] * self.station_ub for k in K),
                                      names=(f'number_b_{k}' for k in K))

    def add_n_b_constraints(self, K: Iterable):
        logger.debug('Adding n - b constraints (b <= n).')
        return self.m.add_constraints((self.b[k] <= self.n[k] for k in K), names=(f'number_n_{k}' for k in K))

    def add_max_queue_constraints(self, s: int, K: Iterable):
        logger.debug('Adding max queue constraints (allocated cars <= max queue).')
        return self.m.add_constraints(
            (self.m.sum(self.u[s][i, k] for i in self.I[s] if self.reachable[s][i, k]) <= 2 * self.n[k] for k in K),
            names=(f'allocation_2n_{s}_{k}' for k in K))

    def add_allocation_constraints(self, s: int, K: Iterable):
        logger.debug('Adding allocation constraints (every car is allocated to at most one station).')
        return self.m.add_constraints(
            (self.m.sum(self.u[s][i, k] for k in K if self.reachable[s][i, k]) <= 1 for i in self.I[s]),
            names=(f'charger_allocation_{s}_{i}' for i in self.I[s]))

    def add_service_constraint(self, s: int, K: Iterable):
        logger.debug(f'Adding service constraint (min. {self.service_level * 100}% of cars are allocated).')
        return self.m.add_constraint(
            (self.m.sum(self.u[s][i, k] for i in self.I[s] for k in K if self.reachable[s][i, k])
             >= self.service_level * self.l[s]), ctname=f'service_level_{s}')

    def get_build_cost(self, K: Iterable):
        return self.build_cost_param * self.m.sum(self.b[k] for k in K)

    def get_maintenance_cost(self, K: Iterable):
        return self.maintenance_cost_param * self.m.sum(self.n[k] for k in K)

    def get_drive_charge_cost(self, s: int, K: Iterable):
        return self.drive_charge_cost_param * self.m.sum(
            self.u[s][i, k] * self.D[s][i, k] for i in self.I[s] for k in K if self.reachable[s][i, k]
        )

    def get_fixed_charge_cost(self, s: int):
        return self.charge_cost_param * (250 - self.R[s]).sum()

    def set_decision_variables(self, K: Iterable):
        logger.debug(f'We add {2 * len(K)} variables for b and n.')

        self.b = self.create_b(K=K)
        self.n = self.create_n(K=K)

        logger.debug(f'We add {sum(self.reachable[s][:, K].sum() for s in self.S)} variables for u.')
        if self.u is None:
            self.u = []
        for s in self.S:
            created_u = self.create_u_s(s=s, K=K)
            try:
                self.u[s] = created_u
            except:
                self.u.append(created_u)

    def initialize_model(self):
        # check that samples have been added
        if self.S is None:
            raise ValueError('No samples have been added. Please add samples before solving the model.')

        self.m = Model(name='MOPTA - Location Improvement', cts_by_name=True)
        # create decision variables
        logger.info('Creating decision variables...')
        self.set_decision_variables(K=self.J)
        # constraints
        logger.info('Creating constraints...')
        self.set_constraints(K=self.J)
        logger.info(f'Model is initialized.')

    def set_constraints(self, K: Iterable):
        # check whether all constraints have been initialized
        if (self.b_n_constraint is None) != (self.max_queue_constraints == []) != (
                self.allocation_constraints == []) != (self.service_constraints == []):
            raise ValueError(
                'Some constraints have not been initialized. Please initialize all constraints before setting new constraints.')

        # set constrained of fixed number of built chargers
        if self.fixed_station_number is not None:
            if self.fixed_station_number_constraint is None:
                self.fixed_station_number_constraint = self.m.add_constraint(
                    self.m.sum(self.b) == self.fixed_station_number, ctname='fixed_station_number')
            else:
                self.fixed_station_number_constraint = self.m.get_constraint_by_name('fixed_station_number').left_expr \
                    .add(self.m.sum(self.b[k] for k in K))

        if self.b_n_constraint is None:  # then all of them are uninitialized
            self.b_n_constraint = self.add_b_n_constraints(K=K)
            self.n_b_constraint = self.add_n_b_constraints(K=K)
            for s in self.S:
                self.max_queue_constraints.append(self.add_max_queue_constraints(s=s, K=K))
                self.allocation_constraints.append(self.add_allocation_constraints(s=s, K=K))
                self.service_constraints.append(self.add_service_constraint(s=s, K=K))
        else:
            self.b_n_constraint += self.add_b_n_constraints(K=K)
            self.n_b_constraint += self.add_n_b_constraints(K=K)
            for s in self.S:
                self.max_queue_constraints[s] += self.add_max_queue_constraints(s=s, K=K)
                self.service_constraints[s] = self.m.get_constraint_by_name(f'service_level_{s}').left_expr. \
                    add(self.m.sum(self.u[s][i, k] for i in self.I[s] for k in K if self.reachable[s][i, k]))
                for i in self.I[s]:
                    self.allocation_constraints[s][i] = self.m.get_constraint_by_name(
                        f'charger_allocation_{s}_{i}').left_expr.add(
                        self.m.sum(self.u[s][i, k] for k in K if self.reachable[s][i, k]))

    def extract_solution(self, sol: SolveSolution, dtype=float):
        logger.info('Extracting solution.')
        b_sol = np.array(sol.get_value_list(dvars=self.b)).round().astype(dtype)
        n_sol = np.array(sol.get_value_list(dvars=self.n)).round().astype(dtype)

        u_sol = []
        for s in self.S:
            u_sol.append(np.zeros(self.u[s].shape))
            u_sol[s][self.reachable[s]] = np.array(sol.get_value_list(dvars=self.u[s][self.reachable[s]].flatten()))
            u_sol[s] = u_sol[s].round().astype(dtype)
        # round needed for numerical stability (e.g. solution with 0.9999999999999999)
        return b_sol, n_sol, u_sol

    def find_improved_locations(self, built_indices: np.ndarray, u_sol: List):
        # create lists for improved locations and their old indices (used for warmstart)
        improved_locations = []
        location_indices = []
        empty_indices = []

        for j in tqdm(built_indices):
            # find allocated cars and their ranges
            X_allocated = []
            ranges_allocated = []
            for s in self.S:
                indices_cars_s = np.argwhere(
                    u_sol[s][:, j] == 1).flatten()  # indices of allocated cars to specific charger
                X_allocated.append(self.X[s][indices_cars_s])
                ranges_allocated.append(self.R[s][indices_cars_s])

            # combine them
            X_allocated = np.vstack(X_allocated)  # combine all car locations from the different samples
            ranges_allocated = np.hstack(ranges_allocated)  # same for the ranges

            if len(X_allocated) != 0:  # if more than zero cars allocated to built charger
                # append new locations
                improved_locations.append(
                    find_optimal_location(allocated_locations=X_allocated, allocated_ranges=ranges_allocated)
                )
                location_indices.append(j)
            else:
                # charger is built bot no cars are allocated
                empty_indices.append(j)

        # convert lists to numpy arrays
        improved_locations = np.array(improved_locations)
        location_indices = np.array(location_indices)
        empty_indices = np.array(empty_indices)

        return improved_locations, location_indices, empty_indices

    def filter_locations(self,
                         improved_locations: np.ndarray,
                         old_location_indices: np.ndarray,
                         min_distance: float = MOPTA_CONSTANTS['min_distance'],
                         counting_radius: float = MOPTA_CONSTANTS['counting_radius']):
        distances = get_distance_matrix(improved_locations, self.L).min(axis=1)
        build_mask = distances > min_distance
        too_close = np.argwhere(build_mask == False).flatten()

        if len(too_close) == 0:
            logger.debug('No locations are too close to other locations. No filtering needed.')
            return improved_locations, old_location_indices
        else:
            # compute distances to all cars and compute how many are in radius
            distances_cars = get_distance_matrix(improved_locations[too_close], self.car_locations)
            number_cars_in_radius = (distances_cars < counting_radius).sum(axis=1) * MOPTA_CONSTANTS[
                'mu_charging']  # multiply by expected charging prob

            # compute distances to all other chargers, and compute how many are in radius
            distances_chargers = get_distance_matrix(improved_locations[too_close], self.L)
            number_locations_radius = (distances_chargers < counting_radius).sum(axis=1) * 2 * self.station_ub

            # compute probability of adding each new location
            prob = number_cars_in_radius / number_locations_radius
            build_mask[too_close] = np.random.uniform(size=len(too_close)) < prob
            logger.debug(f'The probabilities for building of chargers that are too close to others are {prob}.')

            return improved_locations[build_mask], old_location_indices[build_mask]

    def set_objective(self, K: Iterable):
        # sanity check: all of them are None or all of them are not None
        if (self.build_cost is None) != (self.maintenance_cost is None) != (self.drive_charge_cost is None):
            raise ValueError('All of build_cost, maintenance_cost and drive_charge_cost must be None or all must be not None.')

        if self.build_cost is None:  # So that all of them are None
            self.build_cost = self.get_build_cost(K=K)
            self.maintenance_cost = self.get_maintenance_cost(K=K)
            self.drive_charge_cost = sum(self.get_drive_charge_cost(s=s, K=K) for s in self.S)
            self.fixed_charge_cost = sum(self.get_fixed_charge_cost(s=s) for s in self.S)  # independent of K
        else:
            self.build_cost += self.get_build_cost(K=K)
            self.maintenance_cost += self.get_maintenance_cost(K=K)
            self.drive_charge_cost += sum(self.get_drive_charge_cost(s=s, K=K) for s in self.S)

        # set objective
        self.m.minimize(self.build_cost + self.maintenance_cost
                        + 365 / self.n_samples * self.drive_charge_cost
                        + 365 / self.n_samples * self.fixed_charge_cost)
        logger.debug('Objective set.')

    def check_stable(self, warmstart, epsilon: float = 10e-2):
        objective_warmstart = self.m.kpi_value_by_name(name='total_cost', solution=warmstart)
        if abs(self.objective_values[-1] - objective_warmstart) <= epsilon:
            return True
        else:
            return False

    def update_distances_reachable(self, v: int, improved_locations: np.ndarray, K: Iterable):
        for s in self.S:
            self.D[s] = np.concatenate((self.D[s], get_distance_matrix(self.X[s], improved_locations)),
                                       axis=1)  # add new distances
            new_reachable = np.array([self.D[s][i, k] <= self.R[s][i] for i in self.I[s] for k in K]).reshape(
                self.l[s], v)
            self.reachable[s] = np.concatenate((self.reachable[s], new_reachable), axis=1)

    def construct_mip_start(self, u_sol: List, b_sol: np.ndarray, n_sol: np.ndarray,
                            location_indices: np.ndarray,
                            empty_indices: np.ndarray,
                            v: int, K: Iterable, ):
        b_start = np.concatenate((b_sol, np.zeros(v, dtype=float)))
        n_start = np.concatenate((n_sol, np.zeros(v, dtype=float)))
        u_start = []
        for s in self.S:
            u_start.append(np.concatenate((u_sol[s], np.zeros((self.l[s], v), dtype=float)), axis=1, dtype=float))

        # set new locations to built and copy their old n value
        b_start[K] = 1
        n_start[K] = n_sol[location_indices]
        # set old locations to not built
        b_start[location_indices] = 0
        n_start[location_indices] = 0
        # update u
        for s in self.S:
            for k, j in enumerate(location_indices):
                indices_cars = np.argwhere(u_sol[s][:, j] == 1).flatten()
                for i in indices_cars:
                    u_start[s][i, j] = 0
                    u_start[s][i, K[k]] = 1
                    if not self.reachable[s][i, K[k]]:
                        raise ValueError(f"Car {i} cannot reach location {K[k]}")
        # check whether there are built locations that are empty
        if len(empty_indices) > 0:
            logger.info(f'Found {len(empty_indices)} built locations with no cars allocated ->set them to 0.')
            for i in empty_indices:
                b_start[i] = 0
                n_start[i] = 0

        # construct the MIP start with the arrays computed above
        mip_start = self.m.new_solution()
        # name solution
        mip_start.name = 'warm start'
        for j in self.J:
            if b_start[j] == 1:
                if n_start[j] == 0:
                    logger.warning('Built location with n=0.')
                    continue  # skip built locations with n=0, because b should be set to 0 then

                mip_start.add_var_value(self.b[j], b_start[j])
                mip_start.add_var_value(self.n[j], n_start[j])
        for s in self.S:
            for u_dv, u_val in zip(self.u[s][self.reachable[s]], u_start[s][self.reachable[s]]):
                if u_val == 1:
                    mip_start.add_var_value(u_dv, u_val)

        return mip_start, b_start, n_start, u_start

    def set_kpis(self):
        if self.kpi_total is not None:
            self.m.remove_kpi(self.kpi_total)
        if self.kpi_build is not None:
            self.m.remove_kpi(self.kpi_build)
        if self.kpi_maintenance is not None:
            self.m.remove_kpi(self.kpi_maintenance)
        if self.kpi_drive_charge is not None:
            self.m.remove_kpi(self.kpi_drive_charge)
        if self.kpi_fixed_charge is not None:
            self.m.remove_kpi(self.kpi_fixed_charge)

        # add new kpis
        self.kpi_total = self.m.add_kpi(self.build_cost + self.maintenance_cost
                                        + 365 / self.n_samples * self.drive_charge_cost
                                        + 365 / self.n_samples * self.fixed_charge_cost,
                                        'total_cost')
        self.kpi_build = self.m.add_kpi(self.build_cost, 'build_cost')
        self.kpi_maintenance = self.m.add_kpi(self.maintenance_cost, 'maintenance_cost')
        self.kpi_drive_charge = self.m.add_kpi(365 / self.n_samples * self.drive_charge_cost, 'drive_charge_cost')
        self.kpi_fixed_charge = self.m.add_kpi(365 / self.n_samples * self.fixed_charge_cost, 'fixed_charge_cost')

        logger.debug('KPIs set.')

    def report_kpis(self, solution,
                    kpis=['total_cost', 'fixed_charge_cost', 'build_cost', 'maintenance_cost', 'drive_charge_cost']):
        logger.info(f'KPIs {solution.name}:')
        for kpi in kpis:
            logger.info(f'  - {kpi}: {round(self.m.kpi_value_by_name(name=kpi, solution=solution), 2)}')

    def solve(self,
              epsilon_stable: float = 10e-2,
              counting_radius: float = MOPTA_CONSTANTS['counting_radius'],
              min_distance: float = MOPTA_CONSTANTS['min_distance'],
              timelimit: int = 60,
              verbose: bool = False,
              ):

        # sanity check for at least the number of fixed locations
        if self.fixed_station_number is not None and self.fixed_station_number > self.w:
            raise ValueError('Number of fixed locations is larger than the number of available locations. '
                             'Please add more locations, or reduce the number of fixed locations.')

        # compute all maximum service levels to check for infeasibility
        max_service_levels = [compute_maximum_matching(n=np.repeat(8, self.w), reachable=self.reachable[s]) for s in
                              self.S]
        if min(max_service_levels) < self.service_level:
            raise ValueError('Service level cannot be reached with the given number of locations. '
                             'Please add more locations, or reduce service level')
        logger.debug(f'Maximum service levels for samples: {max_service_levels}')

        # monitor time
        start_time = time.time()

        # initialize model, set objective and kpi's
        self.initialize_model()
        self.set_objective(K=self.J)
        self.set_kpis()

        # set solve parameters
        self.m.parameters.preprocessing.presolve = 0  # turn presolve off to avoid issues after lcoation improvement
        self.m.parameters.mip.limits.solutions = 1  # stop after every found solution

        # start the optimization routine
        logger.info('Starting the optimization routine.')
        # optimise the model without a time limit to get a feasible starting solution
        sol = self.m.solve(log_output=verbose, clean_before_solve=False)
        if self.m.solve_details.status == 'integer infeasible':
            raise ValueError('Model is infeasible. Please add more initial locations and / or '
                             'increase fixed number of chargers (if fixed) and/or reduce service level.')

        # If it is feasible, then solution is found and we can continue from there
        logger.info(f'First feasible solution is found.')

        # set timelimit per improvement iteration for future solves
        self.m.parameters.timelimit.set(timelimit)

        while True:
            # This while loop runs until either
            # - the objective value does not increase
            # - no improved location is found

            # count inner iterations for logging and potential future improvement tracking
            inner_iteration_counter = 0
            while True:  # run this while improvement is good enough
                inner_iteration_counter += 1
                old_objective_value = sol.objective_value  # get the value of the current solution

                # solve the model and get the status of the solution
                sol = self.m.solve(log_output=verbose, clean_before_solve=False)
                status = self.m.solve_details.status  # status of the solution
                obj_value = self.m.objective_value  # objective value of found solution

                if status == 'solution limit exceeded':
                    improved = obj_value < old_objective_value  # bool, whether the objective value improved
                    improvement = round(old_objective_value - obj_value, 2)  # improvement in objective value

                    logger.debug(f'Solution found, which is ${improvement} better. Continue with the next iteration.')

                    # logg infor all 4 iterations
                    if inner_iteration_counter % 4 == 0:
                        logger.info(f'Improving the current solution. Current objective value: ${round(obj_value, 2)}')
                    # keep going since the solution limit was exceeded
                    continue

                # if the model is infeasible then most likely the service level is too high for current locations
                # -> add more locations
                elif status == 'integer infeasible':
                    raise ValueError('Model is infeasible. Please add more initial locations and / or '
                                     'increase fixed number of chargers (if fixed) and/or reduce the service level.')

                elif status == 'time limit exceeded':
                    # Since no improvement was found in the set time we continue with the location improvement
                    logger.info(f'Time limit exceeded. Continue with location improvement.')
                    break

                elif status == 'integer optimal, tolerance':
                    # if an optimal solution is found we can proceed with the location improvement
                    logger.info(f'Optimal solution found. Continue with location improvement.')
                    break
                else:
                    logger.info(f'Status: {status}.')
                    break

            # extract current solution
            sol.name = 'CPLEX solution'  # name solution for KPI reporting
            b_sol, n_sol, u_sol = self.extract_solution(sol=sol, dtype=int)
            self.solutions.append((b_sol, n_sol, u_sol))  # append solution vectors to list of solutions
            self.objective_values.append(sol.objective_value)  # append objective value of current solution

            # if a streamlit callback function was added -> call it
            if self.streamlit_callback is not None:
                self.streamlit_callback(self)

            # determine which stations are built to improve their location
            built_indices, not_built_indices = get_indice_sets_stations(b_sol)
            logger.debug(f'There are {len(built_indices)} built and {len(not_built_indices)} not built locations.')
            # compute for every built location its best location. Return that location and its indice
            improved_locations, location_indices, empty_indices = self.find_improved_locations(
                built_indices=built_indices, u_sol=u_sol)

            # filter locations that are built within a distance of a not built location
            filtered_improved_locations, filtered_old_indices = self.filter_locations(
                improved_locations=improved_locations,
                old_location_indices=location_indices,
                min_distance=min_distance,
                counting_radius=counting_radius,
            )

            # if no new locations found
            v = len(filtered_improved_locations)
            if v == 0:
                logger.info('No new locations found -> stopping the optimization routine.')
                break

            # add improved locations
            self.added_locations.append(filtered_improved_locations)

            # update problem
            K = range(self.w, self.w + v)  # range for new locations
            self.L = np.concatenate((self.L, filtered_improved_locations))  # update locations

            # update distances and reachable
            self.update_distances_reachable(v=v, improved_locations=filtered_improved_locations, K=K)

            # Update number of locations and location range
            self.w += v
            self.J = range(self.w)
            logger.info(
                f'{len(filtered_improved_locations)} improved new locations found. There are now {self.w} locations.')

            # update new decision variables
            logger.info('Updating decision variables.')
            self.set_decision_variables(K=K)

            # update the problem and resolve
            logger.info('Updating constraints.')
            self.set_constraints(K=K)

            ## Update Objective Function Constituent Parts. Note that the charge_cost doesn't change
            logger.info('Updating objective function.')
            self.set_objective(K=K)

            # update kpis
            self.set_kpis()

            # generate new mip start
            # generate start vector for new solution
            mip_start, b_start, n_start, u_start = self.construct_mip_start(u_sol=u_sol, b_sol=b_sol, n_sol=n_sol,
                                                                            location_indices=filtered_old_indices,
                                                                            empty_indices=empty_indices,
                                                                            v=v, K=K)
            # Add mipstart
            self.m.add_mip_start(mip_start, complete_vars=True, effort_level=4, write_level=3)
            # report both solutions
            self.report_kpis(solution=sol)
            self.report_kpis(solution=mip_start)

            # check if solution is stable -> There was no improvement compare to the last iteration
            # If it is stop the algorithm
            if self.check_stable(epsilon=epsilon_stable, warmstart=mip_start):
                logger.info('Solution is stable -> stopping the optimization routine.')
                break

        # clear model to free resources
        self.m.end()

        # cast b_start and n_start to int since they are not longer needed to be floats for warmstarts
        b_start = b_start.astype('int')
        n_start = n_start.astype('int')

        end_time = time.time()

        # Always return the solution with the optimised locations (we dont care how close)
        # compute the best locations without filtering
        logger.info('Computing improved locations without filtering for minimum distance for current allocations.')
        best_locations, _, _ = self.find_improved_locations(
            built_indices=np.argwhere(b_start == 1).flatten(), u_sol=u_start)

        logger.info(f'Optimization finished in {round(end_time - start_time, 2)} seconds.')
        logger.info(f'There are {b_start.sum()} built locations with a total of {n_start.sum()} chargers.')

        return n_start[b_start == 1], best_locations

    def allocation_problem(self, n_iter: int, L_sol: np.ndarray, n_sol: np.ndarray,
                           verbose: bool = False,
                           timelimit: int = 60):
        # initialize model
        objective_values = []  # objective values of all solutions
        service_levels = []  # service levels of all solutions
        mip_gaps = []  # mip gaps of all solutions

        build_maintenance_term = self.maintenance_cost_param * np.sum(n_sol) + self.build_cost_param * len(n_sol)

        w = len(L_sol)
        J = range(w)
        logger.info(f'Starting allocation problem with {n_iter} iterations.')

        # Create model once and then update it
        m_a = Model('Allocation Problem')
        expected_number_cars = int(self.n_cars * MOPTA_CONSTANTS["mu_charging"])

        logger.info("Creating decision variables")
        # create a general u for the expexted number of cars
        u = np.array(
            [m_a.binary_var(name=f'u_{i}_{j}') for i in range(expected_number_cars) for j in J]
        ).reshape(expected_number_cars, w)

        logger.info(f"Decision variables added.")

        # since some decision variables in some samples have no effect -> turn off presolve
        m_a.parameters.preprocessing.presolve = 0
        # set time limit
        m_a.parameters.timelimit.set(timelimit)

        for i in range(n_iter):
            logger.info(f'Allocation iteration {i + 1}/{n_iter}.')
            # clear all constraints from the previous iteration
            m_a.clear_constraints()

            # sample one sample
            ranges, charging_prob, charging = self.get_sample()
            logger.debug("  - Sample generated.")

            # filter for cars that are charging
            ranges = ranges[charging]
            locations = self.car_locations[charging]
            distances = get_distance_matrix(locations, L_sol)
            reachable = (distances.T <= ranges).T

            # compute attainable service level
            logger.debug("  - Checking what service level is attainable.")
            attainable_service_level = compute_maximum_matching(n=n_sol, reachable=reachable)
            service_level = self.service_level if attainable_service_level >= self.service_level else attainable_service_level

            logger.debug(f"  - Attainable service level: {round(attainable_service_level * 100, 2)}% "
                         f"(set to {round(service_level * 100, 2)})")

            # set up ranges for problem
            l = charging.sum()
            I = range(l)

            # check if size of u is sufficient: if not -> extend u
            if l > u.shape[0]:
                # append decision variables onto u
                size = l - u.shape[0]
                new_u = np.array(
                    [m_a.binary_var(name=f'u_{i}_{j}') for i in range(l - size, l) for j in J]
                ).reshape(size, w)
                u = np.concatenate((u, new_u), axis=0)

            u_reachable = np.where(reachable, u[:l, :], 0)  # define u for this sample

            # Add constraints to it
            logger.debug("  - Setting the allocation constraints.")
            m_a.add_constraints((m_a.sum(u_reachable[i, j] for j in J) <= 1 for i in I))  # allocated up to one charger

            logger.debug("  - Setting the 2 * n constraints.")
            m_a.add_constraints(
                (m_a.sum(u_reachable[i, j] for i in I) <= 2 * n_sol[j] for j in J))  # allocated up to 2n

            logger.debug(f'  - Setting the service level constraint to {round(service_level * 100, 2)}%.')
            m_a.add_constraint(m_a.sum(u_reachable) / l >= service_level)

            logger.debug("  - Setting the objective function for the distance minimisation.")
            constant_term = self.charge_cost_param * 356 * (250 - ranges).sum()
            m_a.minimize(365 * self.drive_charge_cost_param * m_a.sum(u_reachable * distances)
                         + build_maintenance_term
                         + constant_term)

            logger.debug('  - Starting the solve process.')
            sol = m_a.solve(log_output=verbose, clean_before_solve=True)

            # report objective values
            objective_value = sol.objective_value
            logger.debug(f'  - Objective value: ${round(objective_value, 2)}')
            logger.debug(f'  - Build cost: ${round(build_maintenance_term, 2)}')
            logger.debug(f'  - Constant term: ${round(constant_term, 2)}')
            logger.debug(f'  - Distance cost: ${round(objective_value - constant_term - build_maintenance_term, 2)}')

            # add values to lists
            objective_values.append(sol.objective_value)
            service_levels.append(service_level)
            mip_gaps.append(m_a.solve_details.gap)

        # Clear model to free resources
        m_a.end()

        # convert to numpy arrays
        objective_values = np.array(objective_values)
        service_levels = np.array(service_levels)
        mip_gaps = np.array(mip_gaps)

        i_infeasible = np.argwhere(service_levels < self.service_level).flatten()
        feasible = np.argwhere(service_levels >= self.service_level).flatten()

        # Result logging
        logger.info(f'Out of {n_iter} samples, {len(feasible)} are feasible.')
        # check that lists are actually not empty
        if len(feasible) != 0:
            logger.info(f'- Mean objective value (feasible): ${np.round(np.mean(objective_values[feasible]), 2)}.')
        if len(i_infeasible) != 0:
            logger.info(
                f'- Mean objective value (infeasible): ${np.round(np.mean(objective_values[i_infeasible]), 2)} with a mean service level '
                f'of {np.round(np.mean(service_levels[i_infeasible]) * 100, 2)}%.')

        return objective_values, service_levels, mip_gaps
