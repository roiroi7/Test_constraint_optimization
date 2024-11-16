import typing as t
import networkx as nx

from .edge import EdgeData, EffectRel, edge_to_dataclass

from .node import (
    Attribute,
    AttributeId,
    Constraint,
    ConstraintId,
    NodeId,
    NodeData,
    is_attribute_id,
    is_constraint_id,
    node_to_dataclass,
)


class SAM_With_Constraints:
    nxGraph: nx.DiGraph
    initial_change_points: set[AttributeId]

    def __init__(self, G: nx.DiGraph, initial_change_points: set[AttributeId]) -> None:
        self.nxGraph = G
        self.initial_change_points = initial_change_points

    def attributes(self) -> t.Iterable[Attribute]:
        for n in self.nxGraph.nodes:
            if is_attribute_id(n):
                yield node_to_dataclass(n, self.nxGraph)

    def constraints(self) -> t.Iterable[Constraint]:
        for n in self.nxGraph.nodes:
            if is_constraint_id(n):
                yield node_to_dataclass(n, self.nxGraph)

    def edges(self) -> t.Iterable[tuple[NodeId, NodeId, EdgeData]]:
        for e in self.nxGraph.edges(data=True):
            yield (e[0], e[1], edge_to_dataclass(e))

    def changing_attribute_set(self) -> set[AttributeId]:
        result = self.initial_change_points.copy()
        for i in self.initial_change_points:
            result = result.union(filter(is_attribute_id, nx.descendants(self.nxGraph, i)))
        return result

    def changing_attributes(self) -> t.Iterable[Attribute]:
        for n in self.changing_attribute_set():
            yield node_to_dataclass(n, self.nxGraph)

    def is_changing_attribute(self, n: AttributeId) -> bool:
        return n in self.changing_attribute_set()

    @t.overload
    def successors_with_weight(self, n: AttributeId) -> t.Iterable[tuple[EffectRel, Constraint]]: ...
    @t.overload
    def successors_with_weight(self, n: ConstraintId) -> t.Iterable[tuple[EffectRel, Attribute]]: ...
    def successors_with_weight(self, n: NodeId) -> t.Iterable[tuple[EffectRel, NodeData]]:
        successor_edges = t.cast(list[tuple[NodeId, NodeId, dict[str, t.Any]]], self.nxGraph.out_edges(n, data=True))
        return map(
            lambda k: (edge_to_dataclass(k), node_to_dataclass(k[1], self.nxGraph)),
            successor_edges,
        )

    @t.overload
    def predecessors_with_weight(self, n: AttributeId) -> t.Iterable[tuple[EffectRel, Constraint]]: ...
    @t.overload
    def predecessors_with_weight(self, n: ConstraintId) -> t.Iterable[tuple[EffectRel, Attribute]]: ...
    def predecessors_with_weight(self, n: NodeId) -> t.Iterable[tuple[EffectRel, NodeData]]:
        predecessor_edges = t.cast(list[tuple[NodeId, NodeId, dict[str, t.Any]]], self.nxGraph.in_edges(n, data=True))
        return map(
            lambda k: (edge_to_dataclass(k), node_to_dataclass(k[0], self.nxGraph)),
            predecessor_edges,
        )

    def _cast_simple_cycle_elems(
        self, ids: tuple[AttributeId, ConstraintId, AttributeId]
    ) -> tuple[Attribute, EffectRel, Constraint, EffectRel]:
        return (
            node_to_dataclass(ids[0], self.nxGraph),
            edge_to_dataclass((ids[0], ids[1], self.nxGraph.edges[ids[0], ids[1]])),
            node_to_dataclass(ids[1], self.nxGraph),
            edge_to_dataclass((ids[1], ids[2], self.nxGraph.edges[ids[1], ids[2]])),
        )

    def simple_cycles(self) -> t.Iterable[list[tuple[Attribute, EffectRel, Constraint, EffectRel]]]:
        cycles_in_ids = t.cast(list[list[NodeId]], nx.simple_cycles(self.nxGraph))
        for cycle_in_ids in cycles_in_ids:
            if len(cycle_in_ids) < 2:
                continue
            if is_constraint_id(cycle_in_ids[0]):
                first = cycle_in_ids.pop(0)
                cycle_in_ids.append(first)
            cycle_in_ids.append(cycle_in_ids[0])
            # now cycle_in_ids is [attribute1, constraint1, attribute2, ..., attribute1]
            yield [
                self._cast_simple_cycle_elems(
                    t.cast(
                        tuple[
                            AttributeId, ConstraintId, AttributeId
                        ],  # list indeed, but casted here for better type information
                        cycle_in_ids[i : i + 3],
                    )
                )
                for i in range(0, len(cycle_in_ids) - 1, 2)
            ]

    def copy_with_nodes(self) -> "SAM_With_Constraints":
        G = nx.DiGraph()
        G.add_nodes_from(self.nxGraph.nodes(data=True))
        return SAM_With_Constraints(G, self.initial_change_points.copy())
