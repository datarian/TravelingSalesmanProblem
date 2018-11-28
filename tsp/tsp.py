import scipy.spatial as sp
import functools
import numpy as np
import pandas as pd

from collections import namedtuple

class Node:
    def __init__(self, node):
        self.node = int(node['node'])
        self.x = int(node['x'])
        self.y = int(node['y'])

    @property
    def coords(self):
        return (self.x, self.y)

@functools.total_ordering
class Edge:
    def __init__(self, node1, node2, tsp_config):
        self.tsp = tsp_config
        self.node1 = node1
        self.node2 = node2
        self.edge = self.tsp.distance_matrix[node1,node2]

    def __repr__(self):
        return "{}: nodes: ({}, {}). Length: {}".format(self.__class__.__name__, self.node1, self.node2, self.edge)

    def __lt__(self, other):
        return self.edge < other.edge

    def __eq__(self, other):
        return self.edge == other.edge

    @property
    def length(self):
        return self.edge



class TravelingSalesmanProblem:
    """Holds the nodes and distance matrix for a traveling salesman problem.
    It also provides methods for calculating the distance between nodes.

    Parameters:
    -----------

    nodes:              a pandas dataframe with columns [node, x, y]
    distance_matrix:    an nxn symmetric matrix with the distances between nodes.
        If not passed, it is built in the constructor.
    distance_metric:
    """

    def __init__(self, nodes = None, distance_metric = sp.distance.euclidean):
        self._nodes = dict([[i, Node(nodes.iloc[i,:])] for i in range(len(nodes))]) if nodes is not None else None
        if nodes is not None:
            self.build_distance_matrix()
        else:
            self._distance_matrix = None
        self._distance_metric = distance_metric

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        if isinstance(nodes, pd.DataFrame):
            self._nodes = dict([[i, Node(nodes.iloc[i,:])] for i in range(len(nodes))])
        else:
            return False

    @property
    def distance_matrix(self):
        if self._distance_matrix is None:
            self.build_distance_matrix()
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, distance_matrix):
        assert isinstance(distance_matrix, np.ndarray)
        self._distance_matrix = distance_matrix

    def calc_distance(self, c1, c2):
        return self._distance_metric([c1.x,c1.y], [c2.x,c2.y])

    def build_distance_matrix(self):
        assert self.nodes is not None
        coords = [self.nodes[i].coords for i in range(len(self.nodes))]
        self._distance_matrix = sp.distance_matrix(coords,coords, p=2)
