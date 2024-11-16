from ...type import Constraint, SAM_With_Constraints


def compute_constraint_inconsistency(sam: SAM_With_Constraints) -> float:
    def _single_constraint_inconsistency(c: Constraint) -> float:
        in_sum = sum(
            map(
                lambda x: abs(x[0].weight * x[1].range) if sam.is_changing_attribute(x[1].id) else 0,
                sam.predecessors_with_weight(c.id),
            )
        )
        out_sum = sum(
            map(
                lambda x: abs(x[0].weight * x[1].range) if sam.is_changing_attribute(x[1].id) else 0,
                sam.successors_with_weight(c.id),
            )
        )
        if in_sum + out_sum == 0:
            return 0
        return max((in_sum - out_sum) / (in_sum + out_sum), 0)

    return sum(map(_single_constraint_inconsistency, sam.constraints()))
