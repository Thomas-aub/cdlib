import networkx as nx
from node2vec import Node2Vec
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=8, window=10, min_count=1, batch_words=4, p=1.0, q=1.0):
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
    # Precompute Probabilities and Generate Walks
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=p, q=q)

    # Train the Model
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Discover Similar Nodes
    similar_nodes = model.wv.most_similar(2)
    print("Similar Nodes:", similar_nodes)

    # Extract embeddings
    embeddings = {node: model.wv[node] for node in graph.nodes()}

    # Convert embeddings to a Pandas DataFrame
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')

    return embeddings_df

def apply_kmeans(embeddings_df, k=5):
    """
    Apply K-means clustering on the embeddings DataFrame.

    Parameters:
    - embeddings_df: Pandas DataFrame containing the embeddings
    - k: Number of clusters

    Returns:
    - labels: Array of cluster labels for each node
    """
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings_df)

    # Add the cluster labels to the DataFrame
    embeddings_df['cluster'] = labels

    return embeddings_df

def main():
    # Create a Graph
    graph = nx.fast_gnp_random_graph(n=100, p=0.5)

    # Generate embeddings
    embeddings_df = generate_embeddings(graph)

    # Optionally, save the DataFrame to a CSV file
    embeddings_df.to_csv('developement_test/embeddings.csv')

    # Apply K-means clustering
    k = 5  # Number of clusters
    clustered_df = apply_kmeans(embeddings_df, k)

    # Print the DataFrame with cluster labels
    print(clustered_df.head())

    # Visualize the clusters
    plt.scatter(clustered_df.iloc[:, 0], clustered_df.iloc[:, 1], c=clustered_df['cluster'], cmap='viridis')
    plt.title('K-means Clustering of Node Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.show()

if __name__ == "__main__":
    main()
