from ...type import SAM_With_Constraints


def compute_changing_cost(sam: SAM_With_Constraints) -> float:
    cost_by_structure: dict[str, float] = {}
    total_cost = 0
    for attr in sam.changing_attributes():
        if attr.structure == "":
            total_cost += attr.cost
        else:
            cost_by_structure[attr.structure] = max(attr.cost, cost_by_structure.get(attr.structure, 0))
    return total_cost + sum(cost_by_structure.values())
