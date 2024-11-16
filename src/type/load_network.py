# pyright: reportMissingTypeArgument=false

import typing as t
import pandas as pd
import pandera as pa
from pandera.typing import Series, Index
import networkx as nx


from .network import SAM_With_Constraints
from .node import (
    AttributeId,
    is_attribute_id,
    attributeTypeList,
    changeTypeList,
    is_constraint_id,
)


class EdgeListSchema(pa.DataFrameModel):
    constraint: Series[str] = pa.Field(nullable=False, coerce=True)
    attribute: Series[str] = pa.Field(nullable=False, coerce=True)
    weight: Series[float] = pa.Field(nullable=False, coerce=True)

    @pa.dataframe_check()
    def is_bipartite(cls, df: pd.DataFrame) -> pd.Series:
        return df["constraint"].map(is_constraint_id) & df["attribute"].map(is_attribute_id)


class AttributeListSchema(pa.DataFrameModel):
    id: Index[str] = pa.Field(nullable=False)
    type: Series[str] = pa.Field(nullable=False, isin=attributeTypeList)
    range: Series[int] = pa.Field(nullable=False, ge=0)
    cost: Series[int] = pa.Field(nullable=True, ge=0, default=0)
    imp: Series[int] = pa.Field(nullable=True, default=0)
    structure: Series[str] = pa.Field(nullable=False)
    change_type: Series[str] = pa.Field(nullable=False, isin=changeTypeList)

    @pa.check("id")
    def is_attribute_id(cls, series: pd.Series) -> pd.Series:
        return series.apply(is_attribute_id)


def load_network(
    attributes: pd.DataFrame,
    edges: pd.DataFrame,
) -> SAM_With_Constraints:
    EdgeListSchema.validate(edges)
    edges["type"] = "effect"
    attributes["structure"] = attributes["structure"].fillna("")
    AttributeListSchema.validate(attributes)
    nxGraph: nx.DiGraph = nx.from_pandas_edgelist(
        edges, "constraint", "attribute", ["weight", "type"], create_using=nx.DiGraph
    )
    nxGraph.add_nodes_from((id, dict(val)) for id, val in attributes.iterrows())
    nxGraph.add_nodes_from(edges["constraint"].unique(), type="constraint")
    initial_change_points = {
        t.cast(AttributeId, id) for id, val in attributes.iterrows() if val["change_type"] == "change"
    }
    return SAM_With_Constraints(nxGraph, initial_change_points)
