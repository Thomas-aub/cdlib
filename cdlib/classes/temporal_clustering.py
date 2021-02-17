import json
from collections import defaultdict
from .named_clustering import NamedClustering
import networkx as nx


class TemporalClustering(object):

    def __init__(self):
        """
        Temporal Communities representation
        """

        self.clusterings = defaultdict(NamedClustering)
        self.current_observation = 0
        self.obs_to_time = {}
        self.time_to_obs = {}
        self.matching = None
        self.matched = None

    def add_matching(self, matching):
        """
        Add a precomputed matching of the communities.

        :param matching: a list of tuples [(Ti_Ca, Tj_Cb, score), ... ].
                Community names needs to satisfy the pattern {tid}_{cid}, where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        """
        self.matching = matching

    def get_observation_ids(self):
        """
        Returns the list of temporal ids for the available clusterings
        :return: a list of temporal ids
        """
        return list(self.time_to_obs.keys())

    def get_clustering_at(self, time):
        """
        Returns the clustering observed at a given time

        :param time: the time of observation
        :return: a Clustering object
        """
        return self.clusterings[self.time_to_obs[time]]

    def add_clustering(self, clustering, time):
        """
        Add to the Temporal Clustering the communities observed at a given time

        :param clustering: a Clustering object
        :param time: time of observation
        """

        self.clusterings[self.current_observation] = clustering
        self.time_to_obs[time] = self.current_observation
        self.obs_to_time[self.current_observation] = time
        self.current_observation += 1

    def get_community(self, cid):
        """
        Returns the nodes within a given temporal community

        :param cid: community id of the form {tid}_{cid}, where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        :return: list of nodes within cid
        """

        t, cid = cid.split("_")
        coms = self.get_clustering_at(int(t))
        return coms.communities[int(cid)]

    def to_json(self):
        """
        Generate a JSON representation of the TemporalClustering object

        :return: a JSON formatted string representing the object
        """

        tcluster = {'clusters': [], 'matchings': None}
        if self.matching is not None:
            tcluster['matchings'] = self.matching
        elif self.matched is not None:
            tcluster['matchings'] = self.matched

        for tid in self.get_observation_ids():
            ct = self.get_clustering_at(tid)
            partition = {"tid": tid, "communities": ct.named_communities, "algorithm": ct.method_name,
                         "params": ct.method_parameters, "overlap": ct.overlap, "coverage": ct.node_coverage}
            tcluster['clusters'].append(partition)

        return json.dumps(tcluster)

    def clustering_stability_trend(self, method):
        """
        Returns the trend for community stability.
        The stability index is computed for temporally adjacent clustering pairs.

        :param method: a comparison score taking as input two Clustering objects (e.g., NMI, NF1, ARI...)
        :return: a list of floats
        """

        stability = []

        for i in range(self.current_observation-1):
            c_i = self.clusterings[i]
            c_j = self.clusterings[i+1]
            stb = method(c_i, c_j)
            stability.append(stb.score)

        return stability

    def has_explicit_match(self):
        """
        Checks if the algorithm provided an explicit match of temporal communities

        :return: a list of tuple [(Ti_Ca, Tj_Cb, score), ... ].
                Community names are assigned following the pattern {tid}_{cid}, where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        """
        if self.matching is not None:
            return True
        else:
            return False

    def get_explicit_community_match(self):
        """
        Return an explicit matching of computed communities (if it exists)

        :return: a list of tuple [(Ti_Ca, Tj_Cb, score), ... ].
                Community names are assigned following the pattern {tid}_{cid}, where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        """
        return self.matching

    def community_matching(self, method, two_sided=False):
        """
        Reconstruct community matches across adjacent observations using a provided similarity function.

        :param method: a set similarity function with co-domain in [0,1] (e.g., Jaccard)
        :param two_sided: boolean.
                            Whether the match has to be applied only from the past to the future (False, default)
                            or even from the future to the past (True)
        :return: a list of tuples [(Ti_Ca, Tj_Cb, score), ... ].
                Community names are assigned following the pattern {tid}_{cid}, where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        """

        if self.matching is not None:
            return self.matching

        lifecycle = []

        for i in range(self.current_observation-1):
            c_i = self.clusterings[i]
            c_j = self.clusterings[i+1]

            for cid_i, com_i in enumerate(c_i.communities):
                name_i = f"{self.obs_to_time[i]}_{cid_i}"
                best_match = []
                best_score = 0
                for cid_j, com_j in enumerate(c_j.communities):
                    name_j = f"{self.obs_to_time[i+1]}_{cid_j}"
                    match = method(com_i, com_j)
                    if match > best_score:
                        best_match = [name_j]
                        best_score = match
                    elif match == best_score:
                        best_match.append(name_j)

                lifecycle.append((name_i, name_j, best_score))

        if two_sided:

            for i in range(self.current_observation-1, 0, -1):
                c_i = self.clusterings[i]
                c_j = self.clusterings[i-1]

                for cid_i, com_i in enumerate(c_i.communities):
                    name_i = f"{self.obs_to_time[i]}_{cid_i}"
                    best_match = []
                    best_score = 0
                    for cid_j, com_j in enumerate(c_j.communities):
                        name_j = f"{self.obs_to_time[i-1]}_{cid_j}"
                        match = method(com_i, com_j)
                        if match > best_score:
                            best_match = [name_j]
                            best_score = match
                        elif match == best_score:
                            best_match.append(name_j)

                    lifecycle.append((name_i, name_j, best_score))

        self.matched = lifecycle

        return lifecycle

    def lifecycle_polytree(self, method=None, two_sided=False):
        """
        Reconstruct the poly-tree representing communities lifecycles using a provided similarity function.

        :param method: a set similarity function with co-domain in [0,1] (e.g., Jaccard)
        :param two_sided: boolean.
                            Whether the match has to be applied only from the past to the future (False, default)
                            or even from the future to the past (True)
        :return: a networkx DiGraph object.
                Nodes represent communities, their ids are assigned following the pattern {tid}_{cid},
                where tid is the time of observation and
                cid is the position of the community within the Clustering object.
        """

        if self.matching is not None:
            lifecycle = self.matching
        else:
            if method is None:
                raise ValueError("method parameter not specified")
            lifecycle = self.community_matching(method, two_sided)

        pt = nx.DiGraph()
        for u, v, w in lifecycle:
            pt.add_edge(u, v, weight=w)

        return pt
