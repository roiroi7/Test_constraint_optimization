# pyright: reportAttributeAccessIssue=information

import dataclasses
import typing as t
from random import random, randint, choice, sample
import deap.algorithms
import deap.base
import deap.creator
import deap.tools


from .constrained_fitness import ConstrainedFitness
from .objective import FitnessValues, compute_fitness_values
from .feasibility.change_propagation import compute_change_propagation_feasibility
from ..type import NodeId, ConstraintId, is_constraint_id, EdgeData, SAM_With_Constraints

# Hyper parameters
N_POPULATION = 100
CROSSOVER_P = 0.9
CROSSOVER_INDPB = 0.5
MUTATION_P = 0.1
MUTATION_INDPB = 0.1
SELECT_REF_POINT_P = 12
MU = 100
LAMBDA = 200
N_GENERATION = 2000


ChromosomeElement = t.Literal[0] | t.Literal[1]
type Chromosome = list[ChromosomeElement]


class OrderOptimizer:
    base_sam: SAM_With_Constraints
    edges: list[tuple[NodeId, NodeId, EdgeData]]
    indices_by_constraint: dict[ConstraintId, list[int]]
    toolbox: deap.base.Toolbox

    def __init__(self, sam: SAM_With_Constraints):
        self.base_sam = sam
        self.edges = list(sam.edges())
        self.indices_by_constraint = {}
        for i in range(len(self.edges)):
            edge = self.edges[i]
            if is_constraint_id(edge[0]):
                if not edge[0] in self.indices_by_constraint:
                    self.indices_by_constraint[edge[0]] = [i]
                else:
                    self.indices_by_constraint[edge[0]].append(i)
        self.toolbox = deap.base.Toolbox()

    def crossover(self, ind1: Chromosome, ind2: Chromosome) -> Chromosome:
        child: list[ChromosomeElement | None] = [ind1[i] if ind1[i] == ind2[i] else None for i in range(len(ind1))]
        for indices in self.indices_by_constraint.values():
            unknown_indices = [i for i in indices if child[i] is None]
            if len(unknown_indices) == 0:
                unknown_indices = indices
            if all(child[i] != 1 for i in indices):
                i = choice(unknown_indices)
                unknown_indices.remove(i)
                child[i] = 1
                ind1[i] = 1
            if all(child[i] != 0 for i in indices):
                i = choice(unknown_indices)
                unknown_indices.remove(i)
                child[i] = 0
                ind1[i] = 0
            for i in unknown_indices:
                if child[i] is None:
                    val = choice(t.cast(list[ChromosomeElement], [0, 1]))
                    ind1[i] = val
        return ind1

    def mutate(self, ind: Chromosome, indpb: float):
        return (self.organize_chromosome(deap.tools.mutFlipBit(ind, indpb)[0]),)

    def organize_chromosome(self, ind: Chromosome) -> Chromosome:
        for indices in self.indices_by_constraint.values():
            has_zero = any(ind[i] == 0 for i in indices)
            has_one = any(ind[i] == 1 for i in indices)
            if not has_one:
                ind[choice(indices)] = 1
            if not has_zero:
                ind[choice(indices)] = 0
        return ind

    def binary_tournament(self, ind1: Chromosome, ind2: Chromosome) -> Chromosome:
        ind1_constraint_violation = sum(ind1.fitness.constraint_violation)
        ind2_constraint_violation = sum(ind2.fitness.constraint_violation)
        if ind1_constraint_violation > 0:
            if ind1_constraint_violation == ind2_constraint_violation:
                return choice((ind1, ind2))
            if ind1_constraint_violation < ind2_constraint_violation:
                return ind1
            return ind2
        if ind2_constraint_violation > 0:
            return ind1
        return choice((ind1, ind2))

    def _register_types(self) -> None:
        deap.creator.create("FitnessOrderObjectives", ConstrainedFitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessOrderObjectives)

        self.toolbox.register("attr_bool", randint, 0, 1)
        self.toolbox.register(
            "individual", deap.tools.initRepeat, deap.creator.Individual, self.toolbox.attr_bool, len(self.edges)
        )
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)

    def _fitness(self, chromosome: Chromosome) -> FitnessValues:
        sam = self.base_sam.copy_with_nodes()
        for invert, edge in zip(chromosome, self.edges):
            if invert:
                sam.nxGraph.add_edge(edge[1], edge[0], **dataclasses.asdict(edge[2]))
            else:
                sam.nxGraph.add_edge(edge[0], edge[1], **dataclasses.asdict(edge[2]))
        return compute_fitness_values(sam)

    def _constraint_violation(self, chromosome: Chromosome) -> tuple[float]:
        sam = self.base_sam.copy_with_nodes()
        for invert, edge in zip(chromosome, self.edges):
            if invert:
                sam.nxGraph.add_edge(edge[1], edge[0], **dataclasses.asdict(edge[2]))
            else:
                sam.nxGraph.add_edge(edge[0], edge[1], **dataclasses.asdict(edge[2]))
        return (compute_change_propagation_feasibility(sam),)

    def _register_hyper_params(self) -> None:
        self.toolbox.register("evaluate", self._fitness)
        self.toolbox.register("constant_violation", self._constraint_violation)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate, indpb=MUTATION_INDPB)
        self.toolbox.register(
            "select",
            deap.tools.selNSGA3,
            ref_points=deap.tools.uniform_reference_points(nobj=5, p=SELECT_REF_POINT_P),
        )

    def optimize(self) -> list[tuple[Chromosome, FitnessValues, tuple[float]]]:
        self._register_types()
        self._register_hyper_params()
        logbook = deap.tools.Logbook()
        logbook.header = ["gen", "nevals"]
        population = [self.organize_chromosome(ind) for ind in self.toolbox.population(n=N_POPULATION)]
        pareto_front = deap.tools.ParetoFront()

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        for ind in population:
            if ind.fitness.constraint_violation is None:
                ind.fitness.constraint_violation = self.toolbox.constant_violation(ind)

        pareto_front.update(population)

        logbook.record(gen=0, nevals=len(invalid_ind))
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, N_GENERATION + 1):
            # Vary the population
            offspring = []
            for _ in range(LAMBDA):
                op_choice = random()
                if op_choice < CROSSOVER_P:  # Apply crossover
                    ind1, ind2, ind3, ind4 = [self.toolbox.clone(i) for i in sample(population, 4)]
                    ind1 = self.toolbox.mate(self.binary_tournament(ind1, ind2), self.binary_tournament(ind3, ind4))
                    del ind1.fitness.values
                    offspring.append(ind1)
                elif op_choice < CROSSOVER_P + MUTATION_P:  # Apply mutation
                    ind = self.toolbox.clone(choice(population))
                    (ind,) = self.toolbox.mutate(ind)
                    del ind.fitness.values
                    offspring.append(ind)
                else:  # Apply reproduction
                    offspring.append(choice(population))

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            for ind in offspring:
                if ind.fitness.constraint_violation is None:
                    ind.fitness.constraint_violation = self.toolbox.constant_violation(ind)

            # Update the hall of fame with the generated individuals
            pareto_front.update(offspring)

            # Select the next generation population
            population[:] = self.toolbox.select(population + offspring, MU)

            logbook.record(gen=gen, nevals=len(invalid_ind))
            print(logbook.stream)

        return list(
            map(lambda x: (t.cast(Chromosome, x), self._fitness(x), self._constraint_violation(x)), pareto_front)
        )
