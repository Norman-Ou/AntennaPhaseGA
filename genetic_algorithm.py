from typing import List
import random
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(
            self,
            fitness_func,
            individual_func,
            population_size=20,
            chromosome_length=5,
            crossover_rate=0.8,
            mutation_rate=0.01,
            generations=100,
            elitism=True,
            log_func=None,
        ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_func = fitness_func
        self.individual_func = individual_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elitism = elitism
        self.log_func = log_func
        self.population = self._initialize_population()

    def is_valid_degree_sequence(self, x: List[int]) -> bool:
        """
        判断一个整数序列是否为合法的图的度数序列（使用 Havel–Hakimi 算法）
        """
        sequence = sorted(x, reverse=True)

        while sequence:
            sequence = [d for d in sequence if d > 0]
            if not sequence:
                return True

            d = sequence.pop(0)
            if d > len(sequence):
                return False

            for i in range(d):
                sequence[i] -= 1
                if sequence[i] < 0:
                    return False

            sequence.sort(reverse=True)

        return True

    def _initialize_population(self):
        population = []
        while len(population) < self.population_size:
            individual = [self.individual_func() for _ in range(self.chromosome_length)]
            
            if self.is_valid_degree_sequence(individual):
                population.append(individual)
        return population

    def _evaluate_population(self):
        # 并行计算适应度
        # with ProcessPoolExecutor() as executor:
        #     fitness_values = list(executor.map(self.fitness_func, self.population))
        fitness_values = [
            self.fitness_func(individual) for individual in self.population
        ]

        # 筛选出合法个体（度数合法且 fitness >= 0）
        valid_individuals = []
        for ind, fit in zip(self.population, fitness_values):
            sll = -fit
            if sll >= 0:
                valid_individuals.append((ind, fit))

        while len(valid_individuals) < self.population_size:
            new_individual = [self.individual_func() for _ in range(self.chromosome_length)]
            if not self.is_valid_degree_sequence(new_individual):
                continue
            fitness = self.fitness_func(new_individual)

            sll = -fitness
            if sll >= 0:
                valid_individuals.append((ind, fit))
        
        return valid_individuals

    def _select(self, evaluated):
        selected = []
        while len(selected) < self.population_size:
            i1, i2 = random.sample(evaluated, 2)
            winner = i1 if i1[1] > i2[1] else i2
            selected.append(winner[0])
        return selected

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.chromosome_length - 1)
            return (
                parent1[:point] + parent2[point:],
                parent2[:point] + parent1[point:]
            )
        return parent1[:], parent2[:]

    def _mutate(self, chromosome):
        return [
            self.individual_func() if random.random() < self.mutation_rate else gene
            for gene in chromosome
        ]

    def run(self):
        pbar = tqdm(range(self.generations), desc="Running GA", dynamic_ncols=True)

        for gen in pbar:
            evaluated = self._evaluate_population()
            # import pdb; pdb.set_trace()
            if not evaluated:
                pbar.set_description(f"Gen {gen}: No valid individuals")
                self.population = self._initialize_population()
                continue

            best = max(evaluated, key=lambda x: x[1])
            best_sll = -best[1]
            best_x = best[0]
            self.log_func(x=best_x, sll=best_sll, generation_count=gen)

            pbar.set_postfix(best_sll=f"{best_sll:.2f}")

            selected = self._select(evaluated)
            next_population = []

            if self.elitism:
                next_population.append(best[0])

            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self._crossover(parent1, parent2)
                next_population.extend([
                    self._mutate(child1),
                    self._mutate(child2)
                ])

            self.population = next_population[:self.population_size]

        final_evaluated = self._evaluate_population()
        if final_evaluated:
            return max(final_evaluated, key=lambda x: x[1])
        else:
            return None, -1

