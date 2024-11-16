from dataclasses import dataclass
import typing as t

from .node import NodeId
from .exception import UnexpectedTypeException

type EdgeType = t.Literal["effect"]


@dataclass
class EffectRel:
    type: t.Literal["effect"]
    weight: int


type EdgeData = EffectRel


def edge_to_dataclass(edge: tuple[NodeId, NodeId, dict[str, t.Any]]) -> EdgeData | t.Never:
    typename = edge[2].get("type", None)
    if typename == "effect":
        return EffectRel(**edge[2])
    raise UnexpectedTypeException(f"Invalid data on link {edge[0]}->{edge[1]} : {edge[2]}")
