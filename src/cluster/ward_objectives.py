from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from .types_non_hierarchical import centroid_result_columns


# src_df: OptimResultSchema
def ward_clusters_by_objectives(src_df: pd.DataFrame):
    n_samples = len(src_df.index)
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(src_df.loc[:, "cost":"loop"])
    clusters = AgglomerativeClustering(compute_distances=True).fit(scaled_samples)
    n_nodes = n_samples * 2 - 1
    samples_in_cluster = defaultdict[int, list[int]](lambda: [])
    for i in range(n_samples):
        samples_in_cluster[i] = {i}
    for i in range(len(clusters.children_)):
        samples_in_cluster[i + n_samples] = [
            *samples_in_cluster[clusters.children_[i][0]],
            *samples_in_cluster[clusters.children_[i][1]],
        ]
    cluster_centers = pd.DataFrame(
        [src_df[src_df.index.isin(samples_in_cluster[i])].loc[:, "cost":"loop"].mean(axis=0) for i in range(n_nodes)]
    )
    nodes_df = pd.DataFrame(columns=["label", "child1", "child2", "distance", "size", *centroid_result_columns])
    nodes_df["label"] = pd.RangeIndex(n_nodes)
    nodes_df["size"] = pd.Series(len(samples_in_cluster[i]) for i in range(n_nodes))
    for i in range(n_samples, n_nodes):
        nodes_df.at[i, "child1"] = clusters.children_[i - n_samples][0]
        nodes_df.at[i, "child2"] = clusters.children_[i - n_samples][1]
        nodes_df.at[i, "distance"] = clusters.distances_[i - n_samples]
    nodes_df.loc[:, "avg_cost":"avg_loop"] = pd.DataFrame(
        scaler.inverse_transform(cluster_centers),
        columns=["avg_cost", "avg_coordinate", "avg_constraint", "avg_conflict", "avg_loop"],
        dtype=float,
    )

    nn = NearestNeighbors(metric="euclidean", n_neighbors=1)
    nn.fit(scaled_samples)
    nearest_idx = nn.kneighbors(cluster_centers, n_neighbors=1, return_distance=False)
    for i in range(n_nodes):
        nodes_df.at[i, "nearest_chromosome"] = src_df.at[nearest_idx[i][0], "chromosome"]
        nodes_df.at[i, "nearest_cost"] = src_df.at[nearest_idx[i][0], "cost"]
        nodes_df.at[i, "nearest_coordinate"] = src_df.at[nearest_idx[i][0], "coordinate"]
        nodes_df.at[i, "nearest_constraint"] = src_df.at[nearest_idx[i][0], "constraint"]
        nodes_df.at[i, "nearest_conflict"] = src_df.at[nearest_idx[i][0], "conflict"]
        nodes_df.at[i, "nearest_loop"] = src_df.at[nearest_idx[i][0], "loop"]

    return nodes_df, clusters
