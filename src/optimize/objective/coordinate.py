from dataclasses import dataclass

from ...utils.math import clamp, sgn

from ...type import EffectRel, Attribute, Constraint, SAM_With_Constraints


def _min_loss(
    sam: SAM_With_Constraints, attribute: Attribute, in_edge: tuple[EffectRel, Constraint], is_positive: bool
) -> float:
    @dataclass
    class Out_Capacity_Data:
        capacity: int
        loss: float
        range: int
        weight: int

    in_capacity = abs(attribute.range * in_edge[0].weight)
    outs = filter(lambda x: sam.is_changing_attribute(x[1].id), sam.successors_with_weight(in_edge[1].id))
    out_capacities = sorted(
        map(
            lambda x: Out_Capacity_Data(
                capacity=abs(x[0].weight * x[1].range),
                loss=(
                    max(-x[1].imp * sgn(x[0].weight * in_edge[0].weight), 0)
                    if is_positive
                    else max(x[1].imp * sgn(x[0].weight * in_edge[0].weight), 0)
                ),
                range=x[1].range,
                weight=x[0].weight,
            ),
            outs,
        ),
        key=lambda x: x.loss / abs(x.weight),
    )
    total_out_capacity = 0
    total_loss = 0.0
    for out_capacity in out_capacities:
        amount = clamp(0, out_capacity.range, abs((in_capacity - total_out_capacity) / out_capacity.weight))
        total_out_capacity += amount * abs(out_capacity.weight)
        total_loss += out_capacity.loss * amount
        if total_out_capacity >= in_capacity:
            break
    return total_loss


def compute_coordinate_difficulty(sam: SAM_With_Constraints) -> float:
    return sum(
        map(
            lambda attribute: min(
                sum(
                    map(
                        lambda constraint: _min_loss(sam, attribute, constraint, True),
                        sam.successors_with_weight(attribute.id),
                    )
                )
                + max(attribute.imp * attribute.range, 0),
                sum(
                    map(
                        lambda constraint: _min_loss(sam, attribute, constraint, False),
                        sam.successors_with_weight(attribute.id),
                    )
                )
                + max(-attribute.imp * attribute.range, 0),
            ),
            sam.changing_attributes(),
        )
    )
