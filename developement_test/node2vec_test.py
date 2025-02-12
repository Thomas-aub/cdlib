import networkx as nx
from node2vec import Node2Vec
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cdlib.algorithms.internal import DER
from cdlib import NodeClustering
from cdlib.utils import (
    convert_graph_formats,
    __from_nx_to_graph_tool,
    affiliations2nodesets,
    nx_node_integer_mapping,
)


def generate_embeddings(
        g_original: object,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200, 
        workers: int = 8, 
        window: int = 10, 
        min_count: int = 1, 
        batch_words: int = 4, 
        p: float = 1.0, 
        q: float = 1.0
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

    Returns:
    - embeddings_df: Pandas DataFrame containing the embeddings
    """

    #TODO: Ajouter numérotation avec méthode de /algorithms/crisp_partition.py -> markov_clustering()

    g = convert_graph_formats(g_original, nx.Graph)
    g, maps = nx_node_integer_mapping(g)

    if maps is not None:
        matrix = nx.to_scipy_sparse_array(g, nodelist=range(len(maps)))
    else:
        matrix = nx.to_scipy_sparse_array(g)

# Fin TODO


    # Precompute Probabilities and Generate Walks
    node2vec = Node2Vec(g_original, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)

    # Train the Model
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Discover Similar Nodes
    similar_nodes = model.wv.most_similar(2)
    print("Similar Nodes:", similar_nodes)

    # Extract embeddings
    embeddings = {node: model.wv[node] for node in g_original.nodes()}

    # Convert embeddings to a Pandas DataFrame
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')


    # Optionally, save the DataFrame to a CSV file
    embeddings_df.to_csv('developement_test/embeddings.csv')

    # Apply K-means clustering
    k = 5  # Number of clusters

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings_df)

    # Add the cluster labels to the DataFrame
    embeddings_df['cluster'] = labels

     # Print the DataFrame with cluster labels
    print(embeddings_df.head())

    # Visualize the clusters
    plt.scatter(embeddings_df.iloc[:, 0], embeddings_df.iloc[:, 1], c=embeddings_df['cluster'], cmap='viridis')
    plt.title('K-means Clustering of Node Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.show()

    return embeddings_df



def node2vec_kmeans():
    # Create a Graph
    graph = nx.fast_gnp_random_graph(n=100, p=0.5)

    # Generate embeddings
    generate_embeddings(graph)

    

   

    # L'objectif est ensuite de retourner un NodeClustering (voir markov_clustering)
    # Il faut tester qu'on est bien le même résultat de communauté entre l'utilisation de Louvain et celui de node2vec_kmeans

    # Quand on utilise Node2Vec sur des graphes dont les noeuds ne sont pas numérotés cela pose soit un problème soit il est impossible 
    # d'ensuite savoir à quel noeud appartiens chaque Vecteur (le kmeans n'as donc plus de sens). Il faut donc commencer par les numéroter 
    # comme cela est fait en haut, et ensuite on veut retourner un graph avec un attribut communauté.
    # On peut essayer d'utiliser Karaté_CLub pour comparer le clustering fait avec Loyvain de celui fait avec node2vec_kmeans.

if __name__ == "__main__":
    node2vec_kmeans()
