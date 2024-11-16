import typing as t
import argparse
from pathlib import PurePath

import pandas as pd
import matplotlib.pyplot as plt

from .utils import plot_dendrogram
from .ward_objectives import ward_clusters_by_objectives


def add_args_aggromerative(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("--out-dendrogram", dest="out_dendrogram", help="Path of output dendrogram image")


def main_aggromerative(src_df: pd.DataFrame, args: argparse.Namespace):
    dendrogram_path = t.cast(str | None, args.out_dendrogram)
    out_path_obj = PurePath(args.out)
    if dendrogram_path is None:
        dendrogram_path = str(out_path_obj.with_stem(out_path_obj.stem + "_dendrogram").with_suffix(".png"))

    cluster_nodes, clusters = ward_clusters_by_objectives(src_df)
    cluster_nodes.to_csv(args.out, index=False)

    plot_dendrogram(clusters)
    plt.savefig(dendrogram_path)
