from cdlib import algorithms
import random
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cdlib.algorithms.internal import DER
from cdlib import NodeClustering
from cdlib.utils import (
    convert_graph_formats,
    nx_node_integer_mapping,
)
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

import sys
from pathlib import Path

sys.path.append(str(Path("/home/tom/Documents/Code/cdlib/cdlib/viz")))
from networks import plot_network_clusters

def node2vec_kmeans(
        g_original: object,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        workers: int = 4,
        window: int = 10,
        min_count: int = 1,
        batch_words: int = 4,
        p: float = 1.0,
        q: float = 1.0,
        n_clusters: int = 5,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42
    ) -> NodeClustering:
    """
    Generate embeddings for the nodes in the graph using Node2Vec and return them as a Pandas DataFrame.

    Parameters:
    - graph: NetworkX graph
    - dimensions: Dimension of the embeddings
    - walk_length: Length of each random walk
    - num_walks: Number of random walks per node
    - workers: Number of parallel workers
    - window: Window size for the Skip-Gram model
    - min_count: Minimum count of a node to be included in the model
    - batch_words: Number of words (nodes) to process in each batch
    - p: Return parameter (return_hyperparam)
    - q: In-out parameter (neighbor_hyperparam)
    - n_clusters: Number of clusters to form
    - init: Method for initialization of centroids
    - n_init: Number of time the k-means algorithm will be run with different centroid seeds
    - max_iter: Maximum number of iterations of the k-means algorithm for a single run
    - tol: Relative tolerance with regards to inertia to declare convergence
    - random_state: Seed for the random number generator

    Returns:
    - NodeClustering object
    """

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    if maps is not None:
        matrix = nx.to_scipy_sparse_array(g, nodelist=range(len(maps)))
    else:
        matrix = nx.to_scipy_sparse_array(g)

    # Precompute Probabilities and Generate Walks
    node2vec = Node2Vec(g_original, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)

    # Train the Model
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Extract embeddings
    embeddings = {node: model.wv[node] for node in g_original.nodes()}

    # Convert embeddings to a Pandas DataFrame
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')

    

    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    """""
    # Réduction de dimension à 2D avec t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_reduced = tsne.fit_transform(embeddings_df)
    labels = kmeans.fit_predict(embeddings_reduced)

    """""
    labels = kmeans.fit_predict(embeddings_df)
    

    """""
    clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(embeddings_df)
    """""

    # Create a list of communities based on KMeans labels
    communities = [[] for _ in range(n_clusters)]
    for node, label in zip(embeddings_df.index, labels):
        communities[label].append(node)

    # Create a NodeClustering object
    node_clustering = NodeClustering(
        communities=communities,
        graph=g_original,
        method_name="Node2Vec + KMeans",
        method_parameters={
            "dimensions": dimensions,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "workers": workers,
            "window": window,
            "min_count": min_count,
            "batch_words": batch_words,
            "p": p,
            "q": q,
            "n_clusters": n_clusters,
            "init": init,
            "n_init": n_init,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state
        }
    )

    return node_clustering

def create_graph_with_independent_sets():
    
    # Nombre total de nœuds et nombre de groupes
    n = 40
    k = 10  # Nombre d'ensembles indépendants
    nodes_per_group = n // k  # 5 nœuds par groupe

    # Création du graphe vide
    G = nx.Graph()

    # Ajout des nœuds
    groups = {i: list(range(i * nodes_per_group, (i + 1) * nodes_per_group)) for i in range(k)}

    for group in groups.values():
        G.add_nodes_from(group)

    # Ajout des arêtes entre groupes pour relier les ensembles indépendants
    for i in range(k):
        for j in range(i + 1, k):
            for u in groups[i]:
                for v in groups[j]:
                    if u % nodes_per_group == v % nodes_per_group:  # Relie quelques nœuds entre groupes
                        G.add_edge(u, v)
    return G


if __name__ == "__main__":
    # Generate a graph with independent sets
    g = create_graph_with_independent_sets()

    

    # Apply Markov Clustering
    markov_clusters = algorithms.markov_clustering(g)

    # Apply Node2Vec + KMeans clustering
    node2vec_clusters = node2vec_kmeans(g)

    # Apply Louvain Clustering
    louvain_clusters = algorithms.louvain(g)

    # Visualize Node2Vec + KMeans clusters
    position = nx.spring_layout(g)
    plot_network_clusters(
        graph=g,
        partition=node2vec_clusters,
        position=position,
        figsize=(10, 10),
        interactive=True,
        output_file="node2vec_kmeans_clusters.html"
    )

    # Visualize Markov Clustering
    plot_network_clusters(
        graph=g,
        partition=markov_clusters,
        position=position,
        figsize=(10, 10),
        interactive=True,
        output_file="markov_clusters.html"
    )

    # Visualize Louvain Clustering
    plot_network_clusters(
        graph=g,
        partition=louvain_clusters,
        position=position,
        figsize=(10, 10),
        interactive=True,
        output_file="louvain_clusters.html"
    )


    print("Visualizations saved as 'node2vec_kmeans_clusters.html' and 'markov_clusters.html'.")