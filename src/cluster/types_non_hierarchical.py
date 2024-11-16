from dataclasses import dataclass
import typing as t

import pandas as pd
import pandera as pa
import pandera.typing as pat

N_CLUSTERS: t.Final[int] = 4

centroid_result_columns: t.Final[list[str]] = [
    "avg_cost",
    "avg_coordinate",
    "avg_constraint",
    "avg_conflict",
    "avg_loop",
    "nearest_chromosome",
    "nearest_cost",
    "nearest_coordinate",
    "nearest_constraint",
    "nearest_conflict",
    "nearest_loop",
]


@dataclass
class ClusteringResult:
    labels: pd.Series
    centroids: pd.DataFrame  # CentroidsResultSchema


class CentroidsResultSchema(pa.DataFrameModel):
    label: pat.Series[pa.Int]
    avg_cost: pat.Series[float]
    avg_coordinate: pat.Series[float]
    avg_constraint: pat.Series[float]
    avg_conflict: pat.Series[float]
    avg_loop: pat.Series[float]
    nearest_chromosome: pat.Series[str]
    nearest_cost: pat.Series[float]
    nearest_coordinate: pat.Series[float]
    nearest_constraint: pat.Series[float]
    nearest_conflict: pat.Series[float]
    nearest_loop: pat.Series[float]
