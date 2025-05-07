import random
import numpy as np


class Individual:
    def __init__(
        self, thresholds=None, initial_mode=None, mutation_rate=0.1, mutation_std=20
    ):
        if thresholds is None:
            # Initialize with 1 random threshold in a reasonable range
            self.thresholds = [random.randint(50, 300)]
        else:
            self.thresholds = list(thresholds)

        if initial_mode is None:
            self.initial_mode = random.choice([0, 1])
        else:
            assert initial_mode in [0, 1], "initial_mode must be 0 or 1"
            self.initial_mode = initial_mode

        self.fitness = 0.0
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std

    def mutate(self):
        # Mutate threshold
        if random.random() < self.mutation_rate:
            noise = int(np.random.normal(0, self.mutation_std))
            self.thresholds[0] = max(0, self.thresholds[0] + noise)

        # Mutate initial mode
        if random.random() < self.mutation_rate:
            self.initial_mode = 1 - self.initial_mode  # flip 0 <-> 1

    def crossover(self, other):
        # Average threshold and randomly pick initial_mode
        new_threshold = [(self.thresholds[0] + other.thresholds[0]) // 2]
        new_initial_mode = random.choice([self.initial_mode, other.initial_mode])
        return Individual(new_threshold, new_initial_mode)

    def __repr__(self):
        return f"Individual(thresh={self.thresholds[0]}, mode={self.initial_mode}, fitness={self.fitness:.2f})"
