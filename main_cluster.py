import typing as t
import pandas as pd
import argparse

from src.cluster import (
    OptimResultSchema,
    add_args_non_hierarchical,
    main_non_hierarchical,
    add_args_aggromerative,
    main_aggromerative,
)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--src", help="Path of optimization result CSV", required=True)
    arg_parser.add_argument("--out", help="Path of output CSV", required=True)
    # add_args_non_hierarchical(arg_parser)
    add_args_aggromerative(arg_parser)

    args = arg_parser.parse_args()

    src_path = t.cast(str, args.src)

    src_df = pd.read_csv(
        src_path,
        dtype={
            "chromosome": str,
            "cost": float,
            "coordinate": float,
            "constraint": float,
            "conflict": float,
            "loop": float,
        },
    )
    validated_df = OptimResultSchema.validate(src_df)

    # main_non_hierarchical(validated_df, args)
    main_aggromerative(src_df, args)
