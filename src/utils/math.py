import typing as t
from annotated_types import SupportsLe


def clamp[T: SupportsLe](min: T, val: T, max: T) -> T:
    if min <= val <= max:
        return val
    if val <= min:
        return min
    return max


@t.overload
def sgn(x: int) -> int: ...
@t.overload
def sgn(x: float) -> float: ...
def sgn(x: int | float) -> int | float:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0
