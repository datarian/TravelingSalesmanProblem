import unittest

import pandas as pd
import numpy as np

from tsp_heuristic import BestInsertion, BestBestInsertion, TspHeuristic
from tsp import TravelingSalesmanProblem


class TestTspHeuristic(unittest.TestCase):

    def test_bestbest_select_next(self):
        distance = np.array([[0., 1., 2., 3., 6.],
                            [1., 0., 1., 2., 5.],
                            [2., 1., 0., 1., 4.],
                            [3., 2., 1., 0., 3.],
                            [6., 5., 4., 3., 0.]])
        nodes = pd.read_csv("../TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[0:5,:]

        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()

        bbi = BestBestInsertion(nodes,distance, t)
        bbi.best_best_insertion(nodes, distance)
        self.assertIsInstance(nodes,pd.DataFrame)


    def test_best_select_next(self):
        distance = np.array([[0., 1., 2., 3., 6.],
                            [1., 0., 1., 2., 5.],
                            [2., 1., 0., 1., 4.],
                            [3., 2., 1., 0., 3.],
                            [6., 5., 4., 3., 0.]])
        nodes = pd.read_csv("../TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[0:4,:]

        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()

        bi = BestInsertion(t)
        bi.calculate_tour()
        self.assertIsInstance(nodes,pd.DataFrame)

    def test_select_available(self):
        t = TravelingSalesmanProblem()
        h = TspHeuristic(nodes= pd.DataFrame([[1,0,0],[2,1,0]], columns=['node','x','y']), distance=np.zeros((6,6)), tsp_config=t)
        h._insert_into_tour(2,1,3)
        h._insert_into_tour(3,4)
        print("\n The tour: \n{}".format(h.tour))
        print("\nFully connected nodes: {}\n".format(h._get_occupied_nodes_in_tour()))
        print("\n Open nodes: {}".format(h._get_first_last_in_tour()))
