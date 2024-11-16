import pandera as pa
from pandera.typing import Series


class OptimResultSchema(pa.DataFrameModel):
    chromosome: Series[str] = pa.Field()
    cost: Series[float] = pa.Field(nullable=False, coerce=True)
    coordinate: Series[float] = pa.Field(nullable=False, coerce=True)
    constraint: Series[float] = pa.Field(nullable=False, coerce=True)
    conflict: Series[float] = pa.Field(nullable=False, coerce=True)
    loop: Series[float] = pa.Field(nullable=False, coerce=True)
