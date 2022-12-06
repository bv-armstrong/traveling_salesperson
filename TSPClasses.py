#!/usr/bin/python3
import copy
import functools
import math
import numpy as np
import random


class TSPSolution:
    def __init__(self, listOfCities):
        self.route = listOfCities
        self.cost = self._cost_of_route()

    # print( [c._index for c in listOfCities] )

    def _cost_of_route(self):
        cost = 0
        last = self.route[0]
        for city in self.route[1:]:
            cost += last.cost_to(city)
            last = city
        cost += self.route[-1].cost_to(self.route[0])
        return cost

    def enumerate_edges(self):
        elist = []
        c1 = self.route[0]
        for c2 in self.route[1:]:
            dist = c1.cost_to(c2)
            if dist == np.inf:
                return None
            elist.append((c1, c2, int(math.ceil(dist))))
            c1 = c2
        dist = self.route[-1].cost_to(self.route[0])
        if dist == np.inf:
            return None
        elist.append((self.route[-1], self.route[0], int(math.ceil(dist))))
        return elist


def name_for_int(num):
    if num == 0:
        return ''
    elif num <= 26:
        return chr(ord('A') + num - 1)
    else:
        return name_for_int((num - 1) // 26) + name_for_int((num - 1) % 26 + 1)


class Scenario:
    HARD_MODE_FRACTION_TO_REMOVE = 0.20  # Remove 20% of the edges

    def __init__(self, city_locations, difficulty, rand_seed):
        self._difficulty = difficulty

        if difficulty == "Normal" or difficulty == "Hard":
            self._cities = [City(pt.x(), pt.y(),
                                 random.uniform(0.0, 1.0)
                                 ) for pt in city_locations]
        elif difficulty == "Hard (Deterministic)":
            random.seed(rand_seed)
            self._cities = [City(pt.x(), pt.y(),
                                 random.uniform(0.0, 1.0)
                                 ) for pt in city_locations]
        else:
            self._cities = [City(pt.x(), pt.y()) for pt in city_locations]

        num = 0
        for city in self._cities:
            # if difficulty == "Hard":
            city.set_scenario(self)
            city.set_index_and_name(num, name_for_int(num + 1))
            num += 1

        # Assume all edges exists except self-edges
        ncities = len(self._cities)
        self._edge_exists = (np.ones((ncities, ncities)) - np.diag(np.ones(ncities))) > 0

        if difficulty == "Hard":
            self.thin_edges()
        elif difficulty == "Hard (Deterministic)":
            self.thin_edges(deterministic=True)

    def get_cities(self):
        return self._cities

    def randperm(self, n):  # isn't there a numpy function that does this and even gets called in Solver?
        perm = np.arange(n)
        for i in range(n):
            randind = random.randint(i, n - 1)
            save = perm[i]
            perm[i] = perm[randind]
            perm[randind] = save
        return perm

    def thin_edges(self, deterministic=False):
        ncities = len(self._cities)
        edge_count = ncities * (ncities - 1)  # can't have self-edge
        num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE * edge_count)

        can_delete = self._edge_exists.copy()

        # Set aside a route to ensure at least one tour exists
        route_keep = np.random.permutation(ncities)
        if deterministic:
            route_keep = self.randperm(ncities)
        for i in range(ncities):
            can_delete[route_keep[i], route_keep[(i + 1) % ncities]] = False

        # Now remove edges until
        while num_to_remove > 0:
            if deterministic:
                src = random.randint(0, ncities - 1)
                dst = random.randint(0, ncities - 1)
            else:
                src = np.random.randint(ncities)
                dst = np.random.randint(ncities)
            if self._edge_exists[src, dst] and can_delete[src, dst]:
                self._edge_exists[src, dst] = False
                num_to_remove -= 1


class City:
    def __init__(self, x, y, elevation=0.0):
        self._x = x
        self._y = y
        self._elevation = elevation
        self._scenario = None
        self._index = -1
        self._name = None

    def set_index_and_name(self, index, name):
        self._index = index
        self._name = name

    def set_scenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        How much does it cost to get from this city to the destination?
        Note that this is an asymmetric cost function.
         
        In advanced mode, it returns infinity when there is no connection.
        </summary> '''
    MAP_SCALE = 1000.0

    def cost_to(self, other_city):

        assert (type(other_city) == City)

        # In hard mode, remove edges; this slows down the calculation...
        # Use this in all difficulties, it ensures INF for self-edge
        if not self._scenario._edge_exists[self._index, other_city._index]:
            return np.inf

        # Euclidean Distance
        cost = math.sqrt((other_city._x - self._x) ** 2 +
                         (other_city._y - self._y) ** 2)

        # For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
        if not self._scenario._difficulty == 'Easy':
            cost += (other_city._elevation - self._elevation)
            if cost < 0.0:
                cost = 0.0  # Shouldn't it cost something to go downhill, no matter how steep??????

        return int(math.ceil(cost * self.MAP_SCALE))

    def __repr__(self) -> str:
        return self._name

    def __eq__(self, o: object) -> bool:
        if type(o) != type(self):
            return False
        # return self._x == o._x and \
        #        self._y == o._y and \
        #        self._elevation == o._elevation and \
        #        self._index == o._index and \
        #        self._name == o._name
        return self._name == o._name


class ReducedCostState:

    # Initializes the data for a new Reduced Cost State.
    # Assumes the first city in list is the starting city
    def initialize(self, cities):
        self.cost_matrix = np.array(
            [[c1.cost_to(c2) if c1 != c2 else math.inf for c2 in cities] for c1 in cities])
        self.cost = 0
        self.rows = np.array(cities)
        self.cols = np.array(cities)
        self.route = [cities[0]]
        self.last_city = cities[0]
        self.complete = False

        self._reduce_matrix()

    # Reduces the cost matrix for a given state
    def _reduce_matrix(self):
        num_rows = self.cost_matrix.shape[0]
        num_cols = self.cost_matrix.shape[1]

        # Reduce rows
        for row in self.cost_matrix:
            min_row = np.amin(row)
            if min_row == math.inf:
                # Not a valid solution, so we don't care about anything else
                self.cost = math.inf
                return
            for col_i in range(num_cols):
                row[col_i] -= min_row
            self.cost += min_row

        # Reduce columns
        for col_i in range(num_cols):
            min_col = np.amin(self.cost_matrix[:, col_i])
            if min_col == math.inf:
                # Not a valid solution, so we don't care about anything else
                self.cost = math.inf
                return
            for row_i in range(num_rows):
                self.cost_matrix[row_i, col_i] -= min_col
            self.cost += min_col

    # Returns a new state with a given city appended to it
    #   (It is significantly faster to create a copy while visiting a city
    #   than it is to copy and then visit)
    def visit(self, city, complete):
        row_i = np.where(self.rows == self.last_city)
        col_i = np.where(self.cols == city)

        # Create a new, empty state, and initialize cost
        new_state = ReducedCostState()
        new_state.cost = self.cost + self.cost_matrix[row_i, col_i]

        # Remove the row matching last visited city
        new_state.rows = np.delete(self.rows, row_i)
        new_state.cost_matrix = np.delete(self.cost_matrix, row_i, 0)

        # Remove the column matching the first visited city
        new_state.cols = np.delete(self.cols, col_i)
        new_state.cost_matrix = np.delete(new_state.cost_matrix, col_i, 1)

        # Add city to route
        new_state.last_city = city
        new_state.route = self.route.copy()
        if complete:
            # Used to mark a complete array
            new_state.complete = True
        else:
            new_state.route.append(city)

        new_state._reduce_matrix()
        return new_state

    def __init__(self):
        self.complete = None
        self.last_city = None
        self.route = None
        self.cols = None
        self.rows = None
        self.cost = None
        self.cost_matrix = None

    # Compares the cost of this state to another state
    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self) -> str:
        return "Route: {}\n" \
               "Cost: {}\n" \
               "Rows: {}\n" \
               "Columns: {}\n" \
               "Matrix: {}\n\n" \
            .format(self.route, self.cost, self.rows, self.cols, self.cost_matrix.__repr__())


# Expands a given state into it's neighbor states
def expand_state(state, num_cities):
    result = []
    # If we have visited every city and just need to return to the start
    complete = (len(state.route) == num_cities)

    # Loop through every city we haven't yet visited
    for col in state.cols:
        # We don't want to return to the start city unless every other city has been visited
        if not complete and col == state.route[0]:
            continue

        new_state = state.visit(col, complete)
        result.append(new_state)

    return result


# Calculates a state's priority
# Returns a decimal between 0 and 2 (smaller number means a higher priority)
# Half is based off of where its cost is between min_cost (initial states reduced cost)
#   and max_cost (initial bssf)
# Half is based off of how many cities are left to be visited
def calc_priority(state, min_cost, max_cost, num_cities):
    return (state.cost - min_cost) / (max_cost - min_cost) + \
           (len(state.rows) / num_cities)
