import csv
from pathlib import PurePath
import typing as t
import pandas as pd
import argparse

from src.optimize import OrderOptimizer

from src.type import load_network

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--attr", help="Path of attribute CSV", required=True)
    arg_parser.add_argument("--edge", help="Path of edge CSV", required=True)
    arg_parser.add_argument("--out", help="Path of output CSV", required=True)
    arg_parser.add_argument("--out-edges", dest="out_edges", help="Path of output edge list CSV")

    args = arg_parser.parse_args()

    attr_path = t.cast(str, args.attr)
    edge_path = t.cast(str, args.edge)

    attr_df = pd.read_csv(
        attr_path,
        index_col="id",
        dtype={
            "id": str,
            "type": str,
            "range": int,
            "cost": int,
            "imp": int,
            "structure": str,
        },
    )
    edge_df = pd.read_csv(edge_path)
    sam = load_network(attr_df, edge_df)

    optim = OrderOptimizer(sam)
    pareto = optim.optimize()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chromosome", *pareto[0][1]._fields, "constraint violation"])
        writer.writerows(map(lambda x: ["".join(map(str, x[0])), *x[1], *x[2]], pareto))

    out_edges_path = args.out_edges
    if out_edges_path is None:
        out_path_obj = PurePath(args.out)
        out_edges_path = str(out_path_obj.with_stem(out_path_obj.stem + "_edges"))

    with open(out_edges_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "weight"])
        writer.writerows(map(lambda x: [x[0], x[1], x[2].weight], sam.edges()))
