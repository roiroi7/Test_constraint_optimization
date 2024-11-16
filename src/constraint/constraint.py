from itertools import combinations
import typing as t

import pandas as pd


class Constraint:
    weight_indices: dict[str, int]  # attr_name -> index (or -1 if weight is fixed)
    edges: list[tuple[str, str, float]]  # [src_attr, dst_attr, weight][]

    def __init__(self, attrs: list[str], dsm: pd.DataFrame, starting_index: int) -> None:
        is_first_attr: bool = True
        index = starting_index

        self.weight_indices = {}
        for attr in attrs:
            if is_first_attr:
                self.weight_indices[attr] = -1
                is_first_attr = False
            else:
                self.weight_indices[attr] = index
                index += 1

        self.edges = []
        for src, dst in combinations(attrs, 2):
            if dsm.at[src, dst] != 0:
                self.edges.append((src, dst, t.cast(float, dsm.at[src, dst])))

    def _get_weight(self, attr: str, weights: t.Sequence[float]):
        if attr not in self.weight_indices:
            raise ValueError(f'Attribute "{attr}" is not in the constraint')
        if self.weight_indices[attr] == -1:
            return 1
        return weights[self.weight_indices[attr]]

    def loss(self, weights: t.Sequence[float]):
        return sum(
            map(
                lambda edge: (self._get_weight(edge[1], weights) / self._get_weight(edge[0], weights) + edge[2]) ** 2,
                self.edges,
            )
        )

    def to_weight_dict(self, weights: t.Sequence[float]):
        return {
            attr: 1 if self.weight_indices[attr] == -1 else weights[self.weight_indices[attr]]
            for attr in self.weight_indices
        }
