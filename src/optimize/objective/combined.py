import typing as t

from ...type.network import SAM_With_Constraints
from .changing_cost import compute_changing_cost
from .coordinate import compute_coordinate_difficulty
from .constraint import compute_constraint_inconsistency
from .conflict import compute_conflict_loss
from .loop import compute_loop_loss


class FitnessValues(t.NamedTuple):
    cost: float
    coordinate: float
    constraint: float
    conflict: float
    loop: float


def compute_fitness_values(sam: SAM_With_Constraints) -> FitnessValues:
    return FitnessValues(
        compute_changing_cost(sam),
        compute_coordinate_difficulty(sam),
        compute_constraint_inconsistency(sam),
        compute_conflict_loss(sam),
        compute_loop_loss(sam),
    )
