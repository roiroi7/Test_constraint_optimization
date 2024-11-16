from math import prod
from ...type import EffectRel, Attribute, Constraint, SAM_With_Constraints


def compute_loop_loss(sam: SAM_With_Constraints) -> float:
    def _single_loop_loss(cycle: list[tuple[Attribute, EffectRel, Constraint, EffectRel]]) -> float:
        if any(not sam.is_changing_attribute(x[0].id) for x in cycle):
            return 0
        loop_multiplier = prod(
            map(
                lambda elem: (
                    abs(elem[0].range * elem[1].weight)
                    / sum(
                        map(
                            lambda out_edge: abs(out_edge[0].weight * out_edge[1].range),
                            sam.successors_with_weight(elem[2].id),
                        )
                    )
                ),
                cycle,
            )
        )
        return loop_multiplier * sum(
            map(
                lambda elem_t: (
                    (
                        sum(
                            map(
                                lambda in_edge: (
                                    abs(in_edge[0].weight * in_edge[1].range)
                                    if sam.is_changing_attribute(in_edge[1].id) and in_edge[1].id != elem_t[0][0].id
                                    else 0
                                ),
                                sam.predecessors_with_weight(elem_t[0][2].id),
                            )
                        )
                        / sum(
                            map(
                                lambda out_edge: (
                                    abs(out_edge[0].weight * out_edge[1].range)
                                    if sam.is_changing_attribute(out_edge[1].id)
                                    else 0
                                ),
                                sam.successors_with_weight(elem_t[0][2].id),
                            )
                        )
                    )
                    + sum(
                        map(
                            lambda cnstr: (
                                0
                                if cnstr[1].id == cycle[elem_t[1] - 1][2].id
                                else sum(
                                    map(
                                        lambda in_edge: (
                                            abs(in_edge[0].weight * in_edge[1].range)
                                            if sam.is_changing_attribute(in_edge[1].id)
                                            else 0
                                        ),
                                        sam.predecessors_with_weight(cnstr[1].id),
                                    )
                                )
                                / sum(
                                    map(
                                        lambda out_edge: (
                                            abs(out_edge[0].weight * out_edge[1].range)
                                            if sam.is_changing_attribute(out_edge[1].id)
                                            else 0
                                        ),
                                        sam.successors_with_weight(cnstr[1].id),
                                    )
                                )
                            ),
                            sam.predecessors_with_weight(elem_t[0][0].id),
                        )
                    )
                )
                + (1 if elem_t[0][0].change_type == "change" else 0),
                zip(cycle, range(len(cycle))),
            )
        )

    return sum(map(_single_loop_loss, sam.simple_cycles()))
