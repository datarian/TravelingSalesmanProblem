import unittest

import pandas as pd
import numpy as np

from tsp.tsp_heuristic import TspHeuristic, ConstructionHeuristic, BestInsertion, BestBestInsertion, ShortestEdge, GreedyLocalSearch, Swap, Translate, Invert, Mixed
from tsp.tsp import TravelingSalesmanProblem


class TestTspHeuristic(unittest.TestCase):

    def test_bestbest_select_next(self):
        nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()
        bbi = BestBestInsertion(t)
        bbi.calculate_cycle()
        print(sum(bbi.cycle))
        print(bbi.get_cycle())

    def test_best_select_next(self):
        nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem()
        t.nodes = nodes
        t.build_distance_matrix()
        bi = BestInsertion(t)
        bi.calculate_cycle()
        print(sum(bi.cycle))
        self.assertEqual(bi.cycle.shape[0],len(bi.get_cycle()))

    def test_select_available(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:])
        h = ConstructionHeuristic(tsp_config=t)
        h._insert_into_cycle(2,1,3)
        h._insert_into_cycle(3,4)
        print("\n The cycle: \n{}".format(h.cycle))
        print("\nFully connected nodes: {}\n".format(h._get_occupied_nodes_in_cycle()))

    def test_shortest_edge(self):
        nodes = pd.read_csv("./TSP_411.txt", sep='\s+', names=['node', 'x', 'y'], header=None).iloc[130:149,:]
        t = TravelingSalesmanProblem(nodes)
        se = ShortestEdge(t)
        se.calculate_cycle()
        print(sum(se.cycle))
        print(se.cycle)
        print(se.get_cycle())


class TestImprovementHeuristics(unittest.TestCase):

    def test_init_ch(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[130:169,:])
        ch_swap = GreedyLocalSearch(t, Swap)
        ch_swap._init_algo(save_steps=False)
        self.assertEqual(set(ch_swap.cycle), set(t.nodes.keys()))

    def test_get_predecessor_successor(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[140:149,:])
        ch_swap = GreedyLocalSearch(t, Swap)
        ch_swap.cycle = np.arange(5)
        print(ch_swap.cycle)
        i, = np.where(ch_swap.cycle == 1)
        print("The index is: {}".format(i))
        print("Found predecessor: {}".format(ch_swap.move._get_predecessor(i)))
        print("Found successor: {}".format(ch_swap.move._get_successor(i)))

    def test_calculated_tour_contains_nodes(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[140:149,:])
        ch_swap = GreedyLocalSearch(t, Swap)
        print("Distance matrix:\n{}".format(t.distance_matrix))
        ch_swap._init_algo(save_steps=True)
        ch_swap.calculate_cycle(save_steps=True)
        print(ch_swap.steps)
        self.assertEqual(set(ch_swap.cycle), set(t.nodes.keys()))

    def test_calculated_tour_decrease_loss(self):
        t = TravelingSalesmanProblem(nodes = pd.read_csv("./TSP_411.txt", sep=r'\s+', names=['node', 'x', 'y'], header=None).iloc[134:145,:])
        ch_swap = GreedyLocalSearch(t, Swap)
        ch_swap.calculate_cycle(save_steps=True)
        print(ch_swap.steps)
