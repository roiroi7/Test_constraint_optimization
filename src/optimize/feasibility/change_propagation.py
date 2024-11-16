import itertools
from ...type import SAM_With_Constraints


def compute_change_propagation_feasibility(sam: SAM_With_Constraints) -> float:
    cost = 0.0
    for c in sam.constraints():
        if (
            len(sam.changing_attribute_set().intersection(a[1].id for a in sam.successors_with_weight(c.id))) > 0
            and len(sam.changing_attribute_set().intersection(a[1].id for a in sam.predecessors_with_weight(c.id))) == 0
        ):
            cost += sum(
                abs(a[1].range * a[0].weight)
                for a in sam.successors_with_weight(c.id)
                if a[1].id in sam.changing_attribute_set()
            ) / sum(
                abs(a[0].weight)
                for a in itertools.chain(sam.successors_with_weight(c.id), sam.predecessors_with_weight(c.id))
            )
    return cost
