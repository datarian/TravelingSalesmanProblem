from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import *
from collections import namedtuple

from tsp import TravelingSalesmanProblem

locs = pd.read_csv("../TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None)


class TestTsp(TestCase):

    def test_set_locations(self):
        tsp = TravelingSalesmanProblem()
        tsp.nodes = locs
        obtained = tsp.nodes
        self.assertIsInstance(obtained, list)

    def test_calc_distance(self):
        tsp = TravelingSalesmanProblem()
        Node = namedtuple("Node", "node x y")
        c1 = Node(node=1, x=0, y=0)
        c2 = Node(node=2, x=1, y=1)
        result = tsp.calc_distance(c1, c2)
        expected = np.sqrt(2)
        self.assertEqual(result, expected)

    def test_build_distance_matrix(self):
        tsp = TravelingSalesmanProblem()
        tsp.nodes = locs
        tsp.build_distance_matrix()
        expected = locs.shape[0]
        result = tsp.distance_matrix
        obtained = result.shape[1]
        self.assertEqual(obtained, expected)
        # Check symmetry
        self.assertTrue((result.transpose() == result).all())
