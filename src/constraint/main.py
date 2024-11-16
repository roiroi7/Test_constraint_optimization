import typing as t
import pandas as pd

from .constraint import Constraint
from .loss import optimize_constraint_weight


def main_constraint_weight(dsm: pd.DataFrame, constraints_df: pd.DataFrame) -> pd.DataFrame:
    constraints: list[Constraint] = []
    n_prev_weights = 0
    for _, row in constraints_df.iterrows():
        attrs: list[str] = []
        for attr, val in row.items():
            if val != 0:
                attrs.append(t.cast(str, attr))
        new_constraint = Constraint(attrs, dsm, n_prev_weights)
        constraints.append(new_constraint)
        n_prev_weights += len(new_constraint.weight_indices) - 1  # 重みのうち1つは固定値

    result = optimize_constraint_weight(constraints, n_prev_weights)
    print(f"誤差: {result.fun}")

    constraint_weights = pd.DataFrame(constraints_df, dtype=float)
    for i in range(len(constraints)):
        weights = constraints[i].to_weight_dict(result.x)
        for attr, w in weights.items():
            constraint_weights.at[i, attr] = w
    return constraint_weights
