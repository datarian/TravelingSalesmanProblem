import unittest

import pandas as pd
import numpy as np

from tsp.tsp_heuristic import BestInsertion, BestBestInsertion, ShortestEdge, TspHeuristic
from tsp.tsp import TravelingSalesmanProblem


class TestTspHeuristic(unittest.TestCase):

    def test_bestbest_select_next(self):
        nodes = pd.read_csv("./TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()
        bbi = BestBestInsertion(t)
        bbi.calculate_tour()
        print(sum(bbi.tour))
        print(bbi.get_tour())

    def test_best_select_next(self):
        nodes = pd.read_csv("./TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()
        bi = BestInsertion(t)
        bi.calculate_tour()
        print(sum(bi.tour))
        self.assertEqual(bi.tour.shape[0],len(bi.get_tour()))

    def test_select_available(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:])
        h = TspHeuristic(tsp_config=t)
        h._insert_into_tour(2,1,3)
        h._insert_into_tour(3,4)
        print("\n The tour: \n{}".format(h.tour))
        print("\nFully connected nodes: {}\n".format(h._get_occupied_nodes_in_tour()))

    def test_shortest_edge(self):
        nodes = pd.read_csv("./TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem(nodes)
        se = ShortestEdge(t)
        se.calculate_tour()
        print(sum(se.tour))
        print(se.tour)
        print(se.get_tour())
