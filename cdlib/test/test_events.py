import unittest
from cdlib import algorithms
from cdlib import LifeCycle
from cdlib import TemporalClustering
from plotly import graph_objects as go
from matplotlib.pyplot import Figure
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import os
from cdlib.viz import (
    plot_flow,
    plot_event_radar,
    plot_event_radars,
    typicality_distribution,
)


class EventTest(unittest.TestCase):
    def test_creation(self):

        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)
        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

        events = LifeCycle(tc)
        events.compute_events("asur")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)
        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

        events = LifeCycle(tc)
        events.compute_events("greene")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

    def test_custom_matching(self):
        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        events.compute_events_with_custom_matching(jaccard, two_sided=True)
        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

    def test_polytree(self):
        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        g = events.polytree()
        self.assertIsInstance(g, nx.DiGraph)

    def test_null_model(self):
        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        cf = events.validate_flow("0_2", "+")
        self.assertIsInstance(cf, dict)

        vf = events.validate_all_flows("+")
        self.assertIsInstance(vf, dict)

    def test_viz(self):

        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")

        fig = plot_flow(events)
        self.assertIsInstance(fig, go.Figure)

        plot_event_radar(events, "0_2", direction="+")
        plt.savefig("radar.pdf")
        os.remove("radar.pdf")

        plot_event_radars(events, "0_2")
        plt.savefig("radars.pdf")
        os.remove("radars.pdf")

        typicality_distribution(events, "+")
        plt.savefig("td.pdf")
        os.remove("td.pdf")


if __name__ == "__main__":
    unittest.main()
