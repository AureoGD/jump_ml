import numpy as np
import torch
from typing import List, Tuple, OrderedDict as OrderedDictType, Dict

# --- Neural Network Weight Helper Functions ---


def flatten_nn_parameters(model: torch.nn.Module) -> np.ndarray:
    """
    Flattens all parameters of a PyTorch model into a single NumPy array.
    """
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])


def unflatten_parameters_to_state_dict(flat_params: np.ndarray,
                                       model_ref: torch.nn.Module) -> OrderedDictType[str, torch.Tensor]:
    """
    Converts a flat NumPy array of parameters back into a PyTorch state_dict.
    Args:
        flat_params (np.ndarray): Flat array of parameters.
        model_ref (torch.nn.Module): A reference model instance to get parameter shapes and names.
    Returns:
        OrderedDict[str, torch.Tensor]: The state dictionary.
    """
    new_state_dict = OrderedDictType()
    current_idx = 0
    for name, param_ref in model_ref.named_parameters():
        num_elements = param_ref.numel()
        shape = param_ref.shape

        # Extract the slice for the current parameter
        param_slice = flat_params[current_idx:current_idx + num_elements]

        # Reshape and convert to tensor
        new_state_dict[name] = torch.from_numpy(param_slice).reshape(shape).float()
        current_idx += num_elements

    if current_idx != flat_params.size:
        raise ValueError(f"Size mismatch: flat_params has {flat_params.size} elements, "
                         f"but model requires {current_idx}.")
    return new_state_dict


class CEMOptimizer:
    """
    Cross-Entropy Method (CEM) for evolving Neural Network parameters.
    """

    def __init__(
            self,
            param_dim: int,
            population_size: int = 50,
            elite_fraction: float = 0.1,
            initial_std_dev: float = 0.1,
            noise_decay_factor: float = 0.995,  # For decaying added noise
            min_std_dev: float = 1e-3,  # Minimum std dev for exploration
            extra_noise_scale: float = 0.01):  # Initial scale of extra noise
        """
        Initializes the CEM optimizer.

        Args:
            param_dim (int): Dimensionality of the parameter vector (e.g., flattened NN weights).
            population_size (int): Number of candidate solutions (individuals) per generation.
            elite_fraction (float): Fraction of the population to select as elites (e.g., 0.1 for top 10%).
            initial_std_dev (float): Initial standard deviation for the Gaussian distribution of parameters.
            noise_decay_factor (float): Factor by which extra_noise_scale decays each generation.
            min_std_dev (float): Minimum value for standard deviations to maintain exploration.
            extra_noise_scale (float): Initial scale of the extra noise added to std_devs to prevent
                                       premature convergence, similar to epsilon in CEM-RL paper.
        """
        if not (0 < elite_fraction <= 1):
            raise ValueError("Elite fraction must be between 0 (exclusive) and 1 (inclusive).")

        self.param_dim = param_dim
        self.population_size = population_size
        self.num_elites = max(1, int(population_size * elite_fraction))  # Ensure at least one elite

        # Distribution parameters (mean and standard deviations for each parameter)
        self.mean_params = np.zeros(param_dim, dtype=np.float32)
        self.std_devs = np.full(param_dim, initial_std_dev, dtype=np.float32)

        self.noise_decay_factor = noise_decay_factor
        self.min_std_dev = min_std_dev
        self.current_extra_noise_scale = extra_noise_scale

        print(
            f"CEMOptimizer initialized: param_dim={param_dim}, pop_size={population_size}, num_elites={self.num_elites}"
        )
        print(f"Initial mean_params: (zeros), Initial std_devs: {initial_std_dev}")

    def set_initial_mean_params(self, initial_model: torch.nn.Module):
        """
        Sets the initial mean parameters from an existing model.
        Call this after __init__ if you have a pre-trained or reference model.
        """
        self.mean_params = flatten_nn_parameters(initial_model)
        print(f"CEMOptimizer: Initial mean_params set from provided model. Shape: {self.mean_params.shape}")

    def sample_population(self) -> List[np.ndarray]:
        """
        Samples a new population of parameter vectors from the current distribution.
        Returns a list of flat NumPy arrays, each representing a parameter vector.
        """
        population = []
        for _ in range(self.population_size):
            # Sample from N(mean_params, diag(std_devs^2))
            # This is equivalent to adding noise scaled by std_devs to the mean
            individual_params = self.mean_params + self.std_devs * np.random.randn(self.param_dim).astype(np.float32)
            population.append(individual_params)
        return population

    def update_distribution(self, evaluated_population: List[Tuple[np.ndarray, float]]):
        """
        Updates the mean and standard deviations of the parameter distribution
        based on the fitness of the evaluated population.

        Args:
            evaluated_population (List[Tuple[np.ndarray, float]]): 
                A list of tuples, where each tuple is (parameter_vector, fitness_score).
        """
        if len(evaluated_population) != self.population_size:
            raise ValueError("Size of evaluated_population must match population_size.")

        # Sort individuals by fitness in descending order (higher fitness is better)
        evaluated_population.sort(key=lambda x: x[1], reverse=True)

        # Select the elites
        elite_individuals = [ind[0] for ind in evaluated_population[:self.num_elites]]

        if not elite_individuals:
            print("Warning: No elite individuals selected. This should not happen if num_elites >= 1.")
            return

        elite_params_array = np.array(elite_individuals, dtype=np.float32)

        # Update mean: average of elite parameters
        self.mean_params = np.mean(elite_params_array, axis=0)

        # Update standard deviations: std dev of elite parameters
        # Add a small epsilon for numerical stability if all elites are identical for some params
        self.std_devs = np.std(elite_params_array, axis=0) + 1e-8

        # Add decaying extra noise to std_devs (as per CEM-RL paper)
        # This helps prevent premature convergence by maintaining exploration
        extra_noise = self.current_extra_noise_scale * np.random.randn(self.param_dim).astype(np.float32)
        # Add noise proportional to current std_devs, or a fixed amount
        # For simplicity, let's add scaled random noise to current std devs
        # Or, as the paper implies, add a decaying extra variance (epsilon * I to covariance)
        # Here, we add decaying noise directly to std_devs.
        self.std_devs += self.current_extra_noise_scale * np.ones_like(self.std_devs)  # Add a base noise

        # Ensure std_devs do not become too small
        self.std_devs = np.maximum(self.std_devs, self.min_std_dev)

        # Decay the extra noise scale for the next generation
        self.current_extra_noise_scale *= self.noise_decay_factor
        self.current_extra_noise_scale = max(self.current_extra_noise_scale,
                                             self.min_std_dev / 10.0)  # Don't let it decay to zero completely

        best_fitness_this_gen = evaluated_population[0][1]
        print(f"CEM Distribution Updated. Best Fitness: {best_fitness_this_gen:.4f}, "
              f"Mean StdDev: {np.mean(self.std_devs):.6f}, "
              f"Current Extra Noise Scale: {self.current_extra_noise_scale:.6f}")

    def get_best_params(self) -> np.ndarray:
        """Returns the current mean parameters, which represent the best estimate."""
        return self.mean_params
