from itertools import chain
import typing as t
from ...type import EffectRel, Attribute, Constraint, SAM_With_Constraints

Conflict_Base_Weight: t.Final[float] = 1


def compute_conflict_loss(sam: SAM_With_Constraints) -> float:
    def _single_attribute_conflict(attr: Attribute) -> float:
        def _single_constraint_conflict(c_edge: tuple[EffectRel, Constraint]) -> float:
            return (
                max(
                    sum(
                        map(
                            lambda pred: (
                                abs(pred[0].weight * pred[1].range / c_edge[0].weight)
                                if sam.is_changing_attribute(pred[1].id)
                                else 0
                            ),
                            sam.predecessors_with_weight(c_edge[1].id),
                        )
                    )
                    - sum(
                        map(
                            lambda succ: (
                                abs(succ[0].weight * succ[1].range / c_edge[0].weight)
                                if sam.is_changing_attribute(succ[1].id)
                                else 0
                            ),
                            filter(lambda succ: succ[1].id != attr.id, sam.successors_with_weight(c_edge[1].id)),
                        )
                    ),
                    0,
                )
                / attr.range
            )

        max_constraint_conflict = 0
        total_conflict = 0
        for current_conflict in chain(
            ((1,) if attr.change_type == "change" else ()),
            map(
                _single_constraint_conflict,
                sam.predecessors_with_weight(attr.id),
            ),
        ):
            total_conflict += min(current_conflict, max_constraint_conflict) + (
                Conflict_Base_Weight if current_conflict != 0 else 0
            )
            max_constraint_conflict = max(current_conflict, max_constraint_conflict)
        return total_conflict

    return sum(map(_single_attribute_conflict, sam.changing_attributes()))
