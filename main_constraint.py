import csv
from pathlib import PurePath
import typing as t
import pandas as pd
import argparse

from src.constraint import main_constraint_weight
from src.optimize import OrderOptimizer

from src.type import load_network

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dsm", help="Path of DSM CSV", required=True)
    arg_parser.add_argument("--constraint", help="Path of unweighted constraint CSV", required=True)
    arg_parser.add_argument("--out", help="Path of output CSV", required=True)

    args = arg_parser.parse_args()

    dsm_path = t.cast(str, args.dsm)
    constraint_path = t.cast(str, args.constraint)

    dsm_df = pd.read_csv(
        dsm_path,
        index_col=0,
    )
    constraint_df = pd.read_csv(constraint_path)

    weight_df = main_constraint_weight(dsm_df, constraint_df)

    weight_df.to_csv(args.out, header=True, index=True)
