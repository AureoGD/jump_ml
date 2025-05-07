import numpy as np


class StateNormalizer:
    def __init__(self, n_features, n_modes, n_th, fc_modes):
        self.n_features = n_features
        self.n_modes = n_modes
        self.n_th = n_th
        self.fc_modes = fc_modes

        # Use online normalization for uncertain features (e.g., velocities, CoM pos)
        self.use_online_norm = np.zeros(self.n_features, dtype=bool)
        self.use_online_norm[[0, 1, 2, 3, 5, 9, 10, 11, 22, 23, 24, 25]] = (
            True  # r_vel, r_pos, th, dth, dq
        )

        # Fixed min/max for static features
        # fmt: off
        self.min_vals = np.array(
            [
                -5, -5,                # r_vel 0 1
                 0,   0,               # r_pos 2 3
                -np.pi,                # th 4
                -5,                    # dth 5
                -0.50, -2.2, -1.1,     # q 6 7 8
                -30, -30, -30,         # dq 9 10 11
                -0.50, -2.2, -1.1,     # qr 12 13 14
                -50.0 , -50.0, -50.0,  # tau 15 16 17
                self.n_modes,          # mode 18 
                self.n_th,             # transition history 19 
                self.fc_modes,         # contact modes 20 
                0,                     # staganation metric 21
                -5, -5,                # b_vel 22 23
                0,   0,                # b_pos 24 25
            ]
        )

        self.max_vals = np.array(
            [
                5, 5,                # r_vel
                100,   2,            # r_pos
                np.pi,               # th
                5,                   # dth
                1.20, 0.50, 1.10,    # q
                30, 30, 30,          # dq
                1.20, 0.50, 1.10,    # qr
                50.0 , 50.0, 50.0,   # tau 
                0,                   # mode
                0,                   # transition history
                0,                   # contact modes
                1,                   # staganation metric
                5, 5,                # b_vel
                100,   2,            # b_pos
            ]
        )
        # fmt: on

        self.max_vals[[12, 13, 14]] *= 1.15
        self.min_vals[[12, 13, 14]] *= 1.15

        # Mean/std for online features
        self.mean = np.zeros(self.n_features)
        self.std = np.ones(self.n_features)

        # Precompute alpha/beta for static features
        self.alpha = np.zeros(self.n_features)
        self.beta = np.zeros(self.n_features)
        self._setup_static()

    def _setup_static(self):
        for i in range(self.n_features):
            if not self.use_online_norm[i]:
                self.alpha[i] = 2.0 / (self.max_vals[i] - self.min_vals[i])
                self.beta[i] = -(self.max_vals[i] + self.min_vals[i]) / (
                    self.max_vals[i] - self.min_vals[i]
                )

    def update(self, x, momentum=0.001):
        for i in range(self.n_features):
            if self.use_online_norm[i]:
                delta = x[i] - self.mean[i]
                self.mean[i] += momentum * delta
                self.std[i] += momentum * (delta**2 - self.std[i])
                self.std[i] = max(self.std[i], 1e-6)

    def normalize(self, x):
        x_norm = np.zeros_like(x)
        for i in range(len(x_norm)):
            if self.use_online_norm[i]:
                x_norm[i] = (x[i] - self.mean[i]) / self.std[i]
            else:
                x_norm[i] = self.alpha[i] * x[i] + self.beta[i]
        return x_norm

    def save(self, path):
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            alpha=self.alpha,
            beta=self.beta,
            use_online_norm=self.use_online_norm,
        )

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.use_online_norm = data["use_online_norm"]
