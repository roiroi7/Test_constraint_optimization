import typing as t
import argparse
from pathlib import PurePath

import pandas as pd

from src.cluster import k_means_clusters_by_objectives, som_clusters_by_objectives


def add_args_non_hierarchical(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("--out-center", dest="out_center", help="Path of output centers CSV")
    arg_parser.add_argument("--out-umatrix", dest="out_umatrix", help="Path of output umatrix image")


def main_non_hierarchical(src_df: pd.DataFrame, args: argparse.Namespace):
    umatrix_path = t.cast(str | None, args.out_umatrix)
    out_path_obj = PurePath(args.out)
    if umatrix_path is None:
        umatrix_path = str(out_path_obj.with_stem(out_path_obj.stem + "_umatrix").with_suffix(".png"))

    # cluster_result = k_means_clusters_by_objectives(src_df)
    cluster_result = som_clusters_by_objectives(src_df, umatrix_path=umatrix_path)

    src_df["label"] = cluster_result.labels
    src_df.to_csv(args.out, index=False)

    out_center_path = t.cast(str | None, args.out_center)
    if out_center_path is None:
        out_center_path = str(out_path_obj.with_stem(out_path_obj.stem + "_center"))

    cluster_result.centroids.to_csv(out_center_path, index=True)
