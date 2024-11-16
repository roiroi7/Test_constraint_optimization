from dataclasses import dataclass
import typing as t

import networkx as nx

from .exception import UnexpectedTypeException


AttributeId = t.NewType("AttributeId", str)


def is_attribute_id(v: str) -> t.TypeGuard[AttributeId]:
    return v.startswith("a")


ConstraintId = t.NewType("ConstraintId", str)


def is_constraint_id(v: str) -> t.TypeGuard[ConstraintId]:
    return v.startswith("c")


type NodeId = AttributeId | ConstraintId


def is_node_id(v: str) -> t.TypeGuard[NodeId]:
    return is_attribute_id(v) or is_constraint_id(v)


type AttributeType = t.Literal["characteristic"] | t.Literal["performance"]
attributeTypeList: list[AttributeType] = ["characteristic", "performance"]

type ChangeType = t.Literal["fix"] | t.Literal["change"] | t.Literal["normal"]
changeTypeList: list[ChangeType] = ["fix", "change", "normal"]


@dataclass
class Attribute:
    id: AttributeId
    type: AttributeType
    range: int
    cost: int
    imp: int
    structure: str
    change_type: ChangeType


type ConstraintType = t.Literal["constraint"]


@dataclass
class Constraint:
    id: ConstraintId
    type: ConstraintType


type NodeType = AttributeType | ConstraintType

type NodeData = Attribute | Constraint


@t.overload
def node_to_dataclass(node_id: AttributeId, G: nx.DiGraph) -> Attribute: ...
@t.overload
def node_to_dataclass(node_id: ConstraintId, G: nx.DiGraph) -> Constraint: ...
def node_to_dataclass(node_id: NodeId, G: nx.DiGraph) -> NodeData | t.Never:
    typename = G.nodes[node_id].get("type", None)
    if typename == "characteristic" or typename == "performance":
        return Attribute(id=t.cast(AttributeId, node_id), **G.nodes[node_id])
    if typename == "constraint":
        return Constraint(id=t.cast(ConstraintId, node_id), **G.nodes[node_id])
    raise UnexpectedTypeException(f"Invalid data on node {node_id[0]} : {node_id[1]}")
