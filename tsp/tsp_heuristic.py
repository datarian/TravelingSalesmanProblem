import numpy as np
import pandas as pd
import collections
import operator
import random

from tsp.tsp import Edge

__all__ = [
    'ConstructionHeuristic',
    'ImprovementHeuristic',
    'BestInsertion',
    'BestBestInsertion',
    'ShortestEdge',
    'GreedyLocalSearch',
    'Swap','Translate', 'Invert', 'Mixed',
    'GreedyLocalSearchSwap',
    'GreedyLocalSearchTranslate',
    'GreedyLocalSearchInvert',
    'GreedyLocalSearchMixed',
    'SimulatedAnnealing',
    'SimulatedAnnealingMetropolis',
    'SimulatedAnnealingHeatBath'
]


class TspHeuristic:
    def __init__(self, tsp_config):
        self.tsp = tsp_config
        self.num_nodes = self.tsp.distance_matrix.shape[0]
        self.l = 0
        self.start = None

    @property
    def nodes(self):
        return self.tsp.nodes

    def calculate_cycle(self):
        raise NotImplementedError()

    def _cycle_finished(self):
        raise NotImplementedError()

    def loss(self, cycle=None):
        raise NotImplementedError()

    def get_cycle(self, tour=None):
        raise NotImplementedError()

    def get_cycle_for_plotting(self, nodes = None):
        """Builds the list of node coordinates in the cycle's sequence.
        If no nodes are passed in, takes the calculated cycle from the instance
        variable self.cycle."""
        if not nodes:
            nodes = self.get_cycle()
        cycle = [self.nodes[i].coords for i in nodes]
        cycle.append(self.nodes[nodes[0]].coords)
        return cycle

    def get_starting_node_for_plotting(self):
        raise NotImplementedError()

    def get_steps(self):
        return NotImplementedError()


############################################################################################
# The three construction heuristics inherit from a common class


class ConstructionHeuristic(TspHeuristic):
    def __init__(self, tsp_config):
        super().__init__(tsp_config)
        self.cycle = np.zeros_like(tsp_config.distance_matrix,dtype=int)
        self.num_edges = self.num_nodes - 1
        self.stopping_criterion = 2*(self.num_nodes-1)# All nodes have 2 edges attached

    def loss(self, cycle=None):
        if self.l == 0:
            if not self._cycle_finished():
             self.calculate_cycle()
            cycle = self.get_cycle()
            for i in range(len(cycle)-1):
                self.l += self.tsp.distance_matrix[cycle[i]][cycle[i+1]]
            self.l += self.tsp.distance_matrix[cycle[-1]][cycle[0]]
        return self.l

    def get_cycle(self, start=False):
        """Builds the cycle in the correct order."""
        # init
        cycle = []
        included = set()
        if not start:
            start = self.start
        cycle.append(start)
        included.add(start)
        current = start

        finished = False
        while not finished:
            possible_nexts = [i for i, x in enumerate(self.cycle[current,:]) if x]
            next = set(possible_nexts).difference(included)
            if len(possible_nexts) == 1 and current != start:
                finished = True
            elif not next:

                finished = True
            else:
                # We're inside the cycle. Remove any nodes already selected (basically, the one we came from)
                insert = next.pop()
                cycle.append(insert)
                included.add(insert)
                current = insert
        return cycle

    def get_cycle_for_plotting(self):
        """Builds the list of node coordinates in the cycle's sequence."""
        nodes = self.get_cycle()
        cycle = [self.nodes[i].coords for i in nodes]
        cycle.append(self.get_starting_node_for_plotting())
        return cycle

    def get_starting_node_for_plotting(self):
        """Finds the starting node of the cycle."""
        return self.nodes[self.start].coords

    def _select_new_node(self, size=1):
        """Randomly selects one or more of the nodes that have no edge attached so far."""
        available = [i for (i,v) in zip(range(self.cycle.shape[0]),self.cycle.sum(axis=1) == 0) if v]
        selected = random.sample(available, size)
        return selected[0] if size == 1 else selected

    def _insert_into_cycle(self, left, new, right=False):
        """Inserts a node into the cycle. If only left is given, the node is appended after left.
        If left and right are given, the new node goes in between left and right."""
        left = int(left)
        new = int(new)
        if not right:
            self.cycle[left][new] = 1
            self.cycle[new][left] = 1
        else:
            self.cycle[left][right] = 0 # break connection
            self.cycle[right][left] = 0
            self.cycle[left][new] = 1 # append new after left
            self.cycle[new][left] = 1
            self.cycle[new][right] = 1 # prepend new before right
            self.cycle[right][new] = 1

    def _check_loop_closed(self, node1, node2):
        """Checks if there is a closed connection between node 1 and node 2."""
        if self.cycle[node1,:].sum() == 0 or self.cycle[node2,:].sum() == 0:
            # one of the two is not connected, so it's impossible we end
            # up with a closed cycle
            return False
        else:
            if self.cycle[node1,:].sum() == 1:
                start = node1
                end = node2
            elif self.cycle[node2,:].sum() == 1:
                start = node2
                end = node1
            else:
                # Both nodes are already fully connected.
                # While this says nothing about a closed
                # loop, we can't insert anyways.
                return True
            cycle = self.get_cycle(start)
            if cycle[-1] == end:
                # We have a connection between start and end
                return True
            else:
                return False

    def _get_occupied_nodes_in_cycle(self):
        """Returns all nodes with two connections (fully connected)"""
        connected = np.where(self.cycle.sum(axis=1) == 2)
        connections = [(i,j) for i in connected[0] for j in np.where(self.cycle[i,:]==True)[0] if i < j]
        return connections

    def _cycle_finished(self):
        # cycle is finished when all nodes except start and end have 2 neighbors
        if np.sum(self.cycle) == self.stopping_criterion:
            return True
        return False

    def _get_last_in_cycle(self):
        """Finds the last node in the cycle."""
        open_nodes = tuple([i for (i,v) in zip(range(self.num_nodes),self.cycle.sum(axis=1) == 1) if v])
        if not open_nodes:
            return False
        else:
            last = set(open_nodes) - set([self.start])
            if len(last) == 1:
                return last.pop()
            else:
                return False


class BestInsertion(ConstructionHeuristic):
    def __init__(self, tsp_config):
        super().__init__(tsp_config)

    def _init_algo(self):
        """ Initializes the best insertion algorithm.
        Selects three random nodes as the starting cycle.
        """
        self.l = 0
        self.cycle = np.zeros_like(self.tsp.distance_matrix)

        # draw 3 random nodes
        start_nodes = self._select_new_node(size=3)
        self._insert_into_cycle(start_nodes[0],start_nodes[1],start_nodes[2])
        self.start = start_nodes[0]

    def calculate_cycle(self):
        """Runs the best insertion algorithm."""

        self._init_algo()

        while not self._cycle_finished():
            try:
                next = self._select_new_node()
            except ValueError:
                print("No more nodes available! {}".format(set(self.tsp.nodes.keys()) - set(self.cycle)))
                print("Number of None values in cycle: {}".format(len([i for i in self.cycle if i is None])))
            left, right = self._calc_delta_loss(next)
            self._insert_into_cycle(left, next, right)

        return self.loss()

    def _calc_delta_loss(self, new_node):
        """Calculates the increase when the new point is inserted between any of the
        existing nodes.
        The returned list's indices can be used to select the left node for insertion, chosen
        where the added distance is minimal.
        """
        deltas = []
        c = self._get_occupied_nodes_in_cycle() # returns coordinate tuples of fully connected nodes
        start = self.start
        end = self._get_last_in_cycle()

        def d(n1,n2):
            return self.tsp.distance_matrix[n1,n2]

        if len(c) > 0:
            for i in range(len(c)):
                deltas.append(d(c[i][0], new_node) + d(new_node,c[i][1]) - d(c[i][0], c[i][1]))
        else: # We are at the start of the algorithm, there are 3 nodes.
            visited = self.get_cycle()
            second = visited[1]
            c = c + [(start, second), (second, end)]
            deltas.append(d(start, new_node)+d(new_node, second) - d(start, second))
            deltas.append(d(second, new_node)+d(new_node, end) - d(second, end))

        #Check between current end and start of cycle
        deltas.append(d(end, new_node) + d(new_node,start) - d(end,start))
        c = c + [(end, False)]
        insert_between = c[np.argmin(deltas)]

        return insert_between


class BestBestInsertion(ConstructionHeuristic):
    def __init__(self, tsp_config):
        super().__init__(tsp_config)

    def _init_algo(self):
        """ Initializes the best-best insertion algorithm.
        Selects one random node to start the cycle.
        """
        self.cycle = np.zeros_like(self.tsp.distance_matrix)
        self.l = 0
        self.start = self._select_new_node()
        next = [operator.itemgetter(0)(n) for n in sorted(enumerate(self.tsp.distance_matrix[self.start,:]), key=operator.itemgetter(1))][1]
        self._insert_into_cycle(self.start,next)

    def calculate_cycle(self, save_steps=False):
        """Runs the best insertion algorithm."""

        self._init_algo()

        while not self._cycle_finished():
            left, next, right = self._select_next()
            self._insert_into_cycle(left, next, right)

        return self.loss()

    def _select_next(self):
        """Finds the next node to insert. Determines the distance of all nodes
        not in the cycle so far to all the nodes already in the cycle.

        Returns:
        A tuple of the format(left, next, right)
        next:   Node number of next to insert
        left:   The node to the left"""
        visited = self.get_cycle()

        available = np.where(self.cycle.sum(axis=1) == 0)[0] # Select available nodes
        candidates = np.where(self.cycle.sum(axis=1) > 0)[0] # Select the nodes already in the cycle
        available_mask = np.ones_like(self.cycle,dtype=bool) # By default, mask everything
        #Unmask where we could possibly insert
        for row in available:
            for col in candidates:
                available_mask[row][col] = False
        # Build masked distance matrix
        masked_distance = np.ma.array(self.tsp.distance_matrix, mask=available_mask,shrink=False)
        # Get numbers of next and left nodes
        next_after = np.where(masked_distance == masked_distance.min())
        left = next_after[1][0]
        next = next_after[0][0]
        # figure out right node
        if not left == int(visited[-1]):
            l_index = visited.index(left)
            right = visited[l_index+1]
        else:
            right = False
        return (left, next, right)

class ShortestEdge(ConstructionHeuristic):
    def __init__(self, tsp_config):
        super().__init__(tsp_config)
        self.edges = sorted([Edge(i,j,self.tsp) for i in range(self.num_nodes)
                                                for j in range(self.num_nodes) if not i == j])
        self.condition_cycle_premature = self.num_nodes*2

    def _init_algo(self):
        self.cycle = np.zeros_like(self.tsp.distance_matrix)
        self.l = 0
        self.start = self.edges[0].node1
        self._insert_into_cycle(self.edges[0].node1,self.edges[0].node2, right=False)

    def _check_constraints(self, new_edge):
        n1 = new_edge.node1
        n2 = new_edge.node2

        # Check node degrees:
        if self.cycle[n1,:].sum() == 2 or self.cycle[n2,:].sum() == 2:
            return False
        # Check prematurely closed loop.
        if self._check_loop_closed(n1,n2):
            return False
        return True

    def calculate_cycle(self):

        self._init_algo()
        edge_stack = self.edges.copy()
        edge_stack.remove(self.edges[0]) # The first edge already inserted

        while not self._cycle_finished():
            for e in edge_stack:
                if self._check_constraints(e):
                    self._insert_into_cycle(e.node1,e.node2,False)
                    edge_stack.remove(e)
        open_nodes = [i for i in range(self.num_edges) if self.cycle[i,:].sum() == 1]
        self.start = open_nodes[0]

        return self.loss()

##################################################################################################
# Improvement Heuristics

# Moves for the greedy local search algorithm
class Move():
    def __init__(self, heuristic):
        self.heuristic = heuristic
        self.i = None
        self.j = None
        self.i_pre = None
        self.j_pre = None
        self.i_suc = None
        self.j_suc = None

    @property
    def cycle(self):
        return self.heuristic.cycle

    def do(self):
        """Executes the move."""
        raise NotImplementedError()

    def _select_nodes_for_move(self, size=1, exclude=[]):
        """Selects one or more nodes in the cycle.

        Parameters:
        size:       How many nodes to select
        exclude:    List of nodes that are not available for selection"""
        available = set(range(self.heuristic.num_nodes)) - set(exclude)
        return list(np.random.choice(list(available), size, replace=False))

    def _d(self, n1, n2):
        node1 = self.cycle[n1]
        node2 = self.cycle[n2]
        return self.heuristic.tsp.distance_matrix[node1][node2]

    def _get_successor(self, node):
        n = self.cycle.index(node)
        if n <= self.heuristic.num_nodes-2:
            index = n+1
        else:
            index = 0
        return index

    def _get_predecessor(self, node):
        n = self.cycle.index(node)
        if n != 0:
            index = n-1
        else:
            index = self.heuristic.num_nodes-1
        return index


class Swap(Move):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def do(self):
        tau = self.cycle.copy()
        self.i, self.j = self._select_nodes_for_move(size=2)
        self.i_pre, self.j_pre = [self._get_predecessor(n) for n in [self.i, self.j]]
        self.i_suc, self.j_suc = [self._get_successor(n) for n in [self.i, self.j]]
        tau[self.i], tau[self.j] = tau[self.j], tau[self.i]

        return tau


class Translate(Move):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def do(self):
        tau = self.cycle.copy()
        self.i, = self._select_nodes_for_move(size=1)
        self.i_pre = self._get_predecessor(self.i)
        self.i_suc = self._get_successor(self.i)
        self.j, = self._select_nodes_for_move(size=1, exclude=[self.i + self.i_suc])
        self.j_pre = self._get_predecessor(self.j)
        self.j_suc = self._get_successor(self.j)

        node_j = tau[self.j]
        tau.remove(node_j)
        tau = tau[:self.i_suc] + [node_j] +tau[self.i_suc:]
        return tau


class Invert(Move):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def do(self):
        tau = self.cycle.copy()
        i, j, i_suc, j_suc = 0, 0, 0, 0
        while i_suc == j or j_suc == i:
            i, j = self._select_nodes_for_move(size=2)
            if i > j:
                j,i = i,j
            i_suc = self._get_successor(i)
            j_suc = self._get_successor(j)
        #tau = tau[:i_suc]+list(reversed(tau[i_suc:j_suc]))+tau[j_suc:]
        tau[i_suc:j_suc] = tau[i_suc:j_suc][::-1]
        return tau


class Mixed(Move):
    def __init__(self, heuristic):
        super().__init__(heuristic)
        self.moves = [
            Swap(heuristic),
            Translate(heuristic),
            Invert(heuristic)]

    def _choose_move(self):
        return np.random.randint(0,3)

    def do(self):
        m = self._choose_move()
        self.current = m
        return self.moves[m].do()


class ImprovementHeuristic(TspHeuristic):
    def __init__(self, tsp_config):
        super().__init__(tsp_config)
        self.cycle = []

    def loss(self, cycle=None):
        l = 0
        if not cycle:
            cycle = self.cycle
        for i in range(len(cycle)-1):
            row = cycle[i] * self.num_nodes
            col = cycle[i+1]
            l += self.tsp.distance_matrix.item(row+col)
        row = cycle[-1] * self.num_nodes
        col = cycle[0]
        l += self.tsp.distance_matrix.item(row+col)
        return l

    def get_cycle(self):
        return self.cycle

    def get_starting_node_for_plotting(self):
        return self.nodes[self.cycle[0]].coords

    def _accept_cycle(self, dl):
        """Decides whether to accept the cycle or not
        based on the difference in loss."""
        if dl <= 0:
            return True
        return False

    def _init_algo(self, save_steps=False):
        # Create a random permutation of the nodes
        self.cycle = list(np.random.choice(list(self.nodes.keys()),size=self.num_nodes,replace=False))
        self.l = self.loss()
        self.finished = False
        if save_steps:
            self.steps = []
            self.steps.append(self.l)

    def _cycle_finished(self):
        return self.finished


class GreedyLocalSearch(ImprovementHeuristic):
    """Performs a greedy local search with the specified move.

    Parameters:
    move:   A child class of Move"""
    def __init__(self, tsp_config, move):
        super().__init__(tsp_config)
        self.stopping_criterion = 10 * self.num_nodes**2
        self.finished = False
        self.move = move(self)

    def calculate_cycle(self, save_steps=False):
        self._init_algo(save_steps)

        iter = 0
        while iter < self.stopping_criterion:
            tau = self.move.do()
            loss_t = self.loss(tau)
            if loss_t <= self.l:
                self.cycle = tau
                self.l = loss_t
            if save_steps:
                self.steps.append(np.amin([loss_t, self.l]))
            iter += 1
        self.finished = True
        return self.loss()

class GreedyLocalSearchSwap(GreedyLocalSearch):
    def __init__(self, tsp_config):
        super().__init__(tsp_config, Swap)


class GreedyLocalSearchTranslate(GreedyLocalSearch):
    def __init__(self, tsp_config):
        super().__init__(tsp_config, Translate)


class GreedyLocalSearchInvert(GreedyLocalSearch):
    def __init__(self, tsp_config):
        super().__init__(tsp_config, Invert)


class GreedyLocalSearchMixed(GreedyLocalSearch):
    def __init__(self, tsp_config):
        super().__init__(tsp_config, Mixed)


class SimulatedAnnealing(ImprovementHeuristic):
    def __init__(self, tsp_config, criterion='metropolis', move = Swap):
        assert(criterion in ['metropolis', 'heatbath'])
        super().__init__(tsp_config)
        self.t_max = 100
        self.t_min = 1
        self.cooling_factor = 0.99
        self.criterion=criterion
        self.max_it = 1000
        self.move = move(self)

    def _cool(self, t):
        return self.t_max*self.cooling_factor**t

    def _accept(self, dl, temp):
        if self.criterion == 'metropolis':
            if dl < 0:
                return True
            else:
                d = np.random.uniform()
                if d < np.exp([-dl/temp]):
                    return True
        elif self.criterion == 'heatbath':
            d = np.random.uniform()
            if d < 1/(1+np.exp([dl/temp])):
                return True
        return False

    def _init_algo(self, save_steps=False):
        # Create a random permutation of the nodes
        self.cycle = list(np.random.choice(list(self.nodes.keys()),size=self.num_nodes,replace=False))
        self.l = self.loss()
        self.finished = False
        if save_steps:
            self.steps = {}

    def _check_equilibrium(self, i):
        if i < self.max_it:
            return False
        return True

    def _create_candidate(self):
        return self.move.do()

    def calculate_cycle(self, save_steps=False):
        self._init_algo(save_steps)

        temp = self.t_max
        t = 0
        while temp > self.t_min:
            i = 0
            losses = []
            while not self._check_equilibrium(i):
                d_new = self._create_candidate()
                new_loss = self.loss(d_new)
                dl = new_loss - self.l
                if self._accept(dl, temp):
                    self.cycle = d_new
                    self.l = self.loss()
                if save_steps:
                    losses.append(new_loss)
                i += 1
            t += 1
            if save_steps:
                self.steps[temp] = {'min': np.amin(losses),
                                    'max': np.amax(losses),
                                    'mean': np.mean(losses)}
            temp = self._cool(t)
        return self.l


class SimulatedAnnealingMetropolis(SimulatedAnnealing):
    def __init__(self, tsp_config, move=Swap):
        super().__init__(tsp_config, criterion='metropolis', move=move)


class SimulatedAnnealingHeatBath(SimulatedAnnealing):
    def __init__(self, tsp_config, move=Swap):
        super().__init__(tsp_config,  criterion='heatbath', move=move)
