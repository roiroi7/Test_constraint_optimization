import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .types_non_hierarchical import N_CLUSTERS, centroid_result_columns, ClusteringResult


# src_df: OptimResultSchema
def k_means_clusters_by_objectives(src_df: pd.DataFrame) -> ClusteringResult:
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(src_df.loc[:, "cost":"loop"])
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    clusters = kmeans.fit(scaled_samples)
    centroids_df = pd.DataFrame(columns=centroid_result_columns)
    centroids_df["label"] = pd.RangeIndex(N_CLUSTERS)
    centroids_df.loc[:, "avg_cost":"avg_loop"] = pd.DataFrame(
        scaler.inverse_transform(clusters.cluster_centers_),
        columns=["avg_cost", "avg_coordinate", "avg_constraint", "avg_conflict", "avg_loop"],
        dtype=float,
    )

    nn = NearestNeighbors(metric="euclidean", n_neighbors=1)
    nn.fit(scaled_samples)
    nearest_idx = nn.kneighbors(clusters.cluster_centers_, n_neighbors=1, return_distance=False)
    for i in range(N_CLUSTERS):
        centroids_df.at[i, "nearest_chromosome"] = src_df.at[nearest_idx[i][0], "chromosome"]
        centroids_df.at[i, "nearest_cost"] = src_df.at[nearest_idx[i][0], "cost"]
        centroids_df.at[i, "nearest_coordinate"] = src_df.at[nearest_idx[i][0], "coordinate"]
        centroids_df.at[i, "nearest_constraint"] = src_df.at[nearest_idx[i][0], "constraint"]
        centroids_df.at[i, "nearest_conflict"] = src_df.at[nearest_idx[i][0], "conflict"]
        centroids_df.at[i, "nearest_loop"] = src_df.at[nearest_idx[i][0], "loop"]

    # CentroidsResultSchema.validate(centroids_df)
    return ClusteringResult(labels=pd.Series(clusters.labels_), centroids=centroids_df)
