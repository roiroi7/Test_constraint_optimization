import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from somoclu import Somoclu

from .types_non_hierarchical import N_CLUSTERS, centroid_result_columns, ClusteringResult


# src_df: OptimResultSchema
def som_clusters_by_objectives(src_df: pd.DataFrame, umatrix_path: str | None = None):
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(src_df.loc[:, "cost":"loop"])
    scaled_samples_df = pd.DataFrame(scaled_samples)
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    som = Somoclu(
        n_rows=16,
        n_columns=24,
    )
    som.train(scaled_samples, epochs=100)
    som.cluster(kmeans)
    if umatrix_path is not None:
        som.view_umatrix(bestmatches=True, labels=src_df.index, filename=umatrix_path)

    sample_labels = pd.Series(som.clusters[som.bmus[i, 1], som.bmus[i, 0]] for i in range(len(src_df.index)))
    src_df["label"] = sample_labels

    centroids_df = pd.DataFrame(columns=centroid_result_columns, index=pd.RangeIndex(N_CLUSTERS))
    # centroids_df["label"] = pd.RangeIndex(N_CLUSTERS)
    centroids_df.index.name = "label"
    centroids_df.loc[:, "avg_cost":"avg_loop"] = (
        src_df.loc[:, ["label", "cost", "coordinate", "constraint", "conflict", "loop"]]
        .groupby("label")
        .mean()
        .rename(
            columns={
                "cost": "avg_cost",
                "coordinate": "avg_coordinate",
                "constraint": "avg_constraint",
                "conflict": "avg_conflict",
                "loop": "avg_loop",
            }
        )
    )
    nn = NearestNeighbors(metric="euclidean", n_neighbors=1)
    for i in range(N_CLUSTERS):
        nn = nn.fit(scaled_samples_df[sample_labels == i])
        nearest_idx = nn.kneighbors(
            centroids_df.loc[i, "avg_cost":"avg_loop"].values.reshape(1, -1), n_neighbors=1, return_distance=False
        )
        idx = scaled_samples_df[sample_labels == i].index[nearest_idx[0][0]]
        centroids_df.at[i, "nearest_chromosome"] = src_df.at[idx, "chromosome"]
        centroids_df.at[i, "nearest_cost"] = src_df.at[idx, "cost"]
        centroids_df.at[i, "nearest_coordinate"] = src_df.at[idx, "coordinate"]
        centroids_df.at[i, "nearest_constraint"] = src_df.at[idx, "constraint"]
        centroids_df.at[i, "nearest_conflict"] = src_df.at[idx, "conflict"]
        centroids_df.at[i, "nearest_loop"] = src_df.at[idx, "loop"]

    return ClusteringResult(labels=sample_labels, centroids=centroids_df)
