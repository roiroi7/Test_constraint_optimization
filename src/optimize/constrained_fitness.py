import typing as t

import deap.base


class ConstrainedFitness(deap.base.ConstrainedFitness):
    constraint_violation: t.Iterable[float]

    def __deepcopy__(self, memo):
        copy_ = super(deap.base.ConstrainedFitness, self).__deepcopy__(memo)
        copy_.constraint_violation = self.constraint_violation
        return copy_

    def __gt__(self, other: t.Self):
        return not self.__le__(other)

    def __ge__(self, other: t.Self):
        return not self.__lt__(other)

    def __le__(self, other: t.Self):
        self_constraint_violation = sum(self.constraint_violation)
        other_constraint_violation = sum(other.constraint_violation)
        if self_constraint_violation > 0:
            return self_constraint_violation <= other_constraint_violation
        if other_constraint_violation > 0:
            return True
        return self.wvalues <= other.wvalues

    def __lt__(self, other: t.Self):
        self_constraint_violation = sum(self.constraint_violation)
        other_constraint_violation = sum(other.constraint_violation)
        if self_constraint_violation > 0:
            return self_constraint_violation < other_constraint_violation
        if other_constraint_violation > 0:
            return True
        return self.wvalues < other.wvalues

    def __eq__(self, other: t.Self):
        self_constraint_violation = sum(self.constraint_violation)
        other_constraint_violation = sum(other.constraint_violation)
        if self_constraint_violation > 0:
            return self_constraint_violation == other_constraint_violation
        if other_constraint_violation > 0:
            return False
        return self.wvalues == other.wvalues

    def __ne__(self, other: t.Self):
        return not self.__eq__(other)

    def dominates(self, other: t.Self):
        self_constraint_violation = sum(self.constraint_violation)
        other_constraint_violation = sum(other.constraint_violation)
        if self_constraint_violation > 0:
            return self_constraint_violation < other_constraint_violation
        if other_constraint_violation > 0:
            return True

        return super(ConstrainedFitness, self).dominates(other)
