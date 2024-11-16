import typing as t

import numpy as np
import pandas as pd
import scipy.optimize

from .constraint import Constraint


def calc_loss(constraints: list[Constraint]) -> t.Callable[[t.Sequence[float]], float]:
    def _inner(weight: t.Sequence[float]) -> float:
        return sum(map(lambda c: c.loss(weight), constraints))

    return _inner


def optimize_constraint_weight(constraints: list[Constraint], dimension: int) -> scipy.optimize.OptimizeResult:
    print("start optimization")
    return scipy.optimize.basinhopping(calc_loss(constraints), [0] * dimension)
