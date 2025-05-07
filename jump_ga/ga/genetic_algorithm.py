import random
import matplotlib.pyplot as plt
from jump_ga.env_ga.multi_env import MultiGAEnv
from jump_ga.env_ga.switch_rule import SwitchRule
from jump_ga.ga.individual import Individual


def evaluate_population(population, env):
    rule_list = [SwitchRule(ind.thresholds, ind.initial_mode) for ind in population]
    env.reset_generation(rule_list)
    fitnesses = env.run_generation(max_steps=1100)

    for ind, fit in zip(population, fitnesses):
        ind.fitness = fit


def select_parents(population, num_parents):
    selected = []
    for _ in range(num_parents):
        competitors = random.sample(population, 3)
        winner = max(competitors, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected


def evolve(pop_size=10, generations=10, mutation_rate=0.2, render=False):
    _render = render
    population = [Individual(mutation_rate=mutation_rate) for _ in range(pop_size)]
    env = MultiGAEnv(n_individuals=pop_size, render=_render)
    best_fitnesses = []

    for gen in range(generations):
        print(f"\n🌱 Generation {gen}")

        evaluate_population(population, env=env)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        best = population[0]
        best_fitnesses.append(best.fitness)
        print(f"Best: {best}")

        # ✅ Elitism: keep top 1
        elite = population[:1]

        # Reproduce from top parents
        parents = select_parents(population, pop_size // 2)
        offspring = []
        while len(offspring) < pop_size - len(elite):
            p1, p2 = random.sample(parents, 2)
            child = p1.crossover(p2)
            child.mutate()
            offspring.append(child)

        population = elite + offspring  # ✅ Keep the best individual

    # 📈 Plot best fitness across generations
    plt.figure()
    plt.plot(best_fitnesses, marker="o")
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ✅ Final report
    final_best = max(population, key=lambda ind: ind.fitness)
    print("\n🏁 Final Best Individual:")
    print(f"Threshold: {final_best.thresholds[0]}")
    print(f"Initial Mode: {final_best.initial_mode}")
    print(f"Fitness: {final_best.fitness:.2f}")


if __name__ == "__main__":
    evolve(pop_size=8, generations=100, render=True)
