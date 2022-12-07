#!/usr/bin/python3
from queue import PriorityQueue

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
#     from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
from TSPClasses import *


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.get_cities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.get_cities()
        num_cities = len(cities)
        count = 0
        bssf = None
        best_cost = math.inf

        start_time = time.time()
        for start_city in cities:                               # O(n) times
            if time.time() - start_time > time_allowance:       # O(1)
                break
            city = start_city
            cities_copy = cities.copy()                         # O(n)
            route = []
            for _ in range(num_cities-1):                       # O(n) times
                if time.time() - start_time > time_allowance:   # O(1)
                    break
                cities_copy.remove(city)                        # O(n)
                next_city = min(cities_copy, key=lambda x: city.cost_to(x))     # O(n)
                if city.cost_to(next_city) == math.inf:
                    break
                route.append(city)                              # O(1)
                city = next_city                                # O(1)
            route.append(city)                                  # O(1)
            if len(route) == num_cities:                        # O(1)
                solution = TSPSolution(route)                   # O(1)
                if solution.cost < best_cost:                   # O(1)
                    best_cost = solution.cost                   # O(1)
                    bssf = solution                             # O(1)

        end_time = time.time()
        results['cost'] = best_cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns>  
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.get_cities()
        num_cities = len(cities)
        solutions_found = 0
        max_queue_size = 0

        start_time = time.time()

        bssf = self.greedy(time_allowance)['soln']                      # O(n^2)
        q = PriorityQueue()

        state = ReducedCostState()
        state.initialize(cities)                                        # O(n^2)
        min_cost = state.cost
        max_cost = bssf.cost
        q.put((calc_priority(state, min_cost, max_cost, num_cities), state))    # O(log n)

        total_states = 1
        pruned_states = 0

        while not q.empty() and time.time() - start_time < time_allowance:  # Runs up to n! times
            if q.qsize() > max_queue_size:                                  # O(1)
                max_queue_size = q.qsize()

            priority, state = q.get()                                       # O(log n)
            if state.cost >= bssf.cost:                                     # O(1)
                pruned_states += 1
                continue

            for neighbor in expand_state(state, num_cities):    # O(n^2), loop runs n times
                total_states += 1

                if neighbor.complete:
                    solution = TSPSolution(neighbor.route)                  # O(n)
                    if solution.cost < bssf.cost:                           # O(1)
                        solutions_found += 1
                        bssf = solution
                elif neighbor.cost < bssf.cost:
                    p = calc_priority(neighbor, min_cost, max_cost, num_cities)     # O(1)
                    q.put((p, neighbor))                                            # O(log n)
                else:
                    pruned_states += 1

        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = solutions_found
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        #
        # v1 = self.fancy_v1(time_allowance)
        # print("V1 time: " + str(v1['time']))
        # print("V1 cost: " + str(v1['cost']))
        # print("\n")

        results = {}
        cities = self._scenario.get_cities()
        num_cities = len(cities)
        solutions_found = 0
        max_queue_size = 0

        start_time = time.time()
        bssf = self.greedy(time_allowance)['soln']
        bssf_route = np.array(bssf.route)
        found_solution = True
        while found_solution and time.time() - start_time < time_allowance:  # If no updates are made, break
            found_solution = False

            # Sort edges based on weight, descending
            edges = sorted([(bssf.route[i].cost_to(bssf.route[(i + 1) % num_cities]), i) for i in range(num_cities)],
                           reverse=True)

            # This loop goes through the edges in route, longest to shortest
            for edge1 in range(num_cities):

                # This loop goes through remaining edges, shortest to longest
                for edge2 in range(edge1+1, num_cities):

                    # For the inner for loop, we will look at all remaining edges and pick the best one, if applicable
                    best_route = None
                    for edge3 in range(edge2+1, num_cities):
                        # i, j, k are the positions that we need to cut the route, and need to be in that order
                        i, j, k = sorted([edges[edge_i][1] for edge_i in [edge1, edge2, edge3]])
                        # Cut and reassemble the route
                        new_route = np.concatenate((bssf_route[:i], bssf_route[j:k], bssf_route[i:j], bssf_route[k:]))
                        sol = TSPSolution(new_route)
                        if sol.cost < bssf.cost:
                            best_route = new_route
                            bssf = sol
                            solutions_found += 1
                            found_solution = True

                    if best_route is not None:
                        bssf_route = best_route
                        solutions_found += 1

        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = solutions_found
        results['soln'] = bssf
        results['max'] = max_queue_size
        return results

    def fancy_v1(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.get_cities()
        num_cities = len(cities)
        solutions_found = 0
        max_queue_size = 0

        start_time = time.time()
        bssf = self.greedy(time_allowance)['soln']
        best_route = np.array(bssf.route)
        keep_running = True
        while keep_running and time.time() - start_time < time_allowance:
            keep_running = False
            for k in range(num_cities, 0, -1):
                for j in range(k - 1, 0, -1):
                    for i in range(j - 1, 0, -1):
                        new_route = np.concatenate((best_route[:i], best_route[j:k], best_route[i:j], best_route[k:]))
                        sol = TSPSolution(new_route)
                        if sol.cost < bssf.cost:
                            best_route = new_route
                            bssf = sol
                            solutions_found += 1
                            keep_running = True
        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = solutions_found
        results['soln'] = bssf
        results['max'] = max_queue_size
        return results

