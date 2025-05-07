import numpy as np


class ModelMatrices:
    def __init__(self):
        self.L0 = 0.4
        self.L1 = 0.45
        self.L2 = 0.5
        self.L3 = 0.39
        self.r = 0.05

        self.m0 = 7
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1

        self.masses = [self.m0, self.m1, self.m2, self.m3]
        self.m = np.sum(self.masses)

        self.I0 = self._initi_inertia_tensor(self.m0, self.r, self.L0)
        self.I1 = self._initi_inertia_tensor(self.m1, self.r, self.L1)
        self.I2 = self._initi_inertia_tensor(self.m2, self.r, self.L2)
        self.I3 = self._initi_inertia_tensor(self.m3, self.r, self.L3)
        arr = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.I3 = arr @ self.I3 @ arr.transpose()
        self.Inertia = [self.I0, self.I1, self.I2, self.I3]

        # Initialize matrices with zeros
        self.J_com = np.zeros((3, 3))
        self.J_com1 = np.zeros((3, 3))
        self.J_com2 = np.zeros((3, 3))
        self.J_com3 = np.zeros((3, 3))
        self.J_toe = np.zeros((3, 3))
        self.J_heel = np.zeros((3, 3))
        self.J_ankle = np.zeros((3, 3))

        # Initialize transformation matrices
        self.HT_com0 = np.zeros((4, 4))
        self.HT_com1 = np.zeros((4, 4))
        self.HT_com2 = np.zeros((4, 4))
        self.HT_com3 = np.zeros((4, 4))
        self.HT_knee = np.zeros((4, 4))
        self.HT_ankle = np.zeros((4, 4))
        self.HT_toe = np.zeros((4, 4))
        self.HT_heel = np.zeros((4, 4))

        self.HT = [self.HT_com0, self.HT_com1, self.HT_com2, self.HT_com3]

        # Apply the constants to the HT_ matrices
        self._initialize_HT_matrices()

    def _initi_inertia_tensor(self, m, r, h):
        Inertia_tensor = np.zeros((3, 3), dtype=np.float64)
        Inertia_tensor[0, 0] = ((m * h**2) / 12) + ((m * r**2) / 4)
        Inertia_tensor[1, 1] = Inertia_tensor[0, 0]
        Inertia_tensor[2, 2] = (m * r**2) / 2

        return Inertia_tensor

    def _initialize_HT_matrices(self):
        """
        Initializes the transformation matrices (HT_) with specific constants.
        """
        # Create a dictionary for the matrices and their properties
        matrices = {
            "HT_com1": [1.0, -1.0],
            "HT_com2": [1.0, -1.0],
            "HT_com3": [1.0, -1.0],
            "HT_toe": [1.0, -1.0],
            "HT_heel": [1.0, -1.0],
            "HT_knee": [1.0, -1.0],
            "HT_ankle": [1.0, -1.0],
        }

        # Set the constant values for each matrix
        for name, values in matrices.items():
            matrix = getattr(self, name)  # Get the matrix by name
            matrix[3, 3] = values[0]  # Set the (3,3) position
            matrix[1, 2] = values[1]  # Set the (1,2) position

        self.HT_com0[0:4, 0:4] = np.identity(4)
        self.HT_com0[2, 3] = self.L0 / 2

    def compute_sin_cos(self, angles):
        """
        Utility function to compute sin and cos for multiple angles at once.
        This avoids redundant computation of sin and cos for the same angles.
        """
        sin_vals = np.sin(angles)
        cos_vals = np.cos(angles)
        return sin_vals, cos_vals

    def update_robot_states(self, q, dq):
        """
        Updates the robot states (q and dq) and recomputes sin and cos for the angles.
        """
        self.q = np.array(q)
        self.dq = np.array(dq)

        # Compute sin and cos values for the updated angles
        self.sin_vals, self.cos_vals = self.compute_sin_cos(
            [self.q[0], self.q[0] + self.q[1], self.q[0] + self.q[1] + self.q[2]]
        )

    def update_kinematics(self):
        self.update_homog_trans()
        self.update_jacobians()

    def update_homog_trans(self):
        # Use the updated sin and cos values from the robot state
        sin_vals = self.sin_vals
        cos_vals = self.cos_vals

        # Update HT_com1
        self.HT_com1[0, 0] = sin_vals[0]
        self.HT_com1[0, 1] = cos_vals[0]
        self.HT_com1[0, 3] = 0.225 * sin_vals[0]
        self.HT_com1[2, 0] = -cos_vals[0]
        self.HT_com1[2, 1] = sin_vals[0]
        self.HT_com1[2, 3] = -0.225 * cos_vals[0]

        # Update HT_com2
        self.HT_com2[0, 0] = sin_vals[1]
        self.HT_com2[0, 1] = cos_vals[1]
        self.HT_com2[0, 3] = 0.45 * sin_vals[0] + 0.25 * sin_vals[1]
        self.HT_com2[2, 0] = -cos_vals[1]
        self.HT_com2[2, 1] = sin_vals[1]
        self.HT_com2[2, 3] = -0.45 * cos_vals[0] - 0.25 * cos_vals[1]

        # Update HT_com3
        self.HT_com3[0, 0] = sin_vals[2]
        self.HT_com3[0, 1] = cos_vals[2]
        self.HT_com3[0, 3] = 0.45 * sin_vals[0] + 0.065 * cos_vals[2] + 0.5 * sin_vals[1]
        self.HT_com3[2, 0] = -cos_vals[2]
        self.HT_com3[2, 1] = sin_vals[2]
        self.HT_com3[2, 3] = 0.065 * sin_vals[2] - 0.45 * cos_vals[0] - 0.5 * cos_vals[1]

        # Update HT_toe
        self.HT_toe[0, 0] = sin_vals[2]
        self.HT_toe[0, 1] = cos_vals[2]
        self.HT_toe[0, 3] = 0.45 * sin_vals[0] + 0.26 * cos_vals[2] + 0.05 * sin_vals[2] + 0.50 * sin_vals[1]
        self.HT_toe[2, 0] = -cos_vals[2]
        self.HT_toe[2, 1] = sin_vals[2]
        self.HT_toe[2, 3] = 0.26 * sin_vals[2] - 0.05 * cos_vals[2] - 0.45 * cos_vals[0] - 0.5 * cos_vals[1]

        # Update HT_heel
        self.HT_heel[0, 0] = sin_vals[2]
        self.HT_heel[0, 1] = cos_vals[2]
        self.HT_heel[0, 3] = 0.45 * sin_vals[0] - 0.13 * cos_vals[2] + 0.05 * sin_vals[2] + 0.50 * sin_vals[1]
        self.HT_heel[2, 0] = -cos_vals[2]
        self.HT_heel[2, 1] = sin_vals[2]
        self.HT_heel[2, 3] = -0.13 * sin_vals[2] - 0.05 * cos_vals[2] - 0.45 * cos_vals[0] - 0.5 * cos_vals[1]

        # Update HT_knee
        self.HT_knee[0, 0] = sin_vals[1]
        self.HT_knee[0, 1] = cos_vals[1]
        self.HT_knee[0, 3] = 0.45 * sin_vals[0]
        self.HT_knee[2, 0] = -cos_vals[1]
        self.HT_knee[2, 1] = sin_vals[1]
        self.HT_knee[2, 3] = -0.45 * cos_vals[0]

        # Update HT_ankle
        self.HT_ankle[0, 0] = sin_vals[2]
        self.HT_ankle[0, 1] = cos_vals[2]
        self.HT_ankle[0, 3] = 0.45 * sin_vals[0] + 0.5 * sin_vals[1]
        self.HT_ankle[2, 0] = -cos_vals[2]
        self.HT_ankle[2, 1] = sin_vals[2]
        self.HT_ankle[2, 3] = -0.45 * cos_vals[0] - 0.5 * cos_vals[1]

    def update_jacobians(self):
        # Use the updated sin and cos values from the robot state
        sin_vals = self.sin_vals
        cos_vals = self.cos_vals

        # Jacobian for HT_toe
        self.J_toe[0, 0] = 0.45 * cos_vals[0] - 0.26 * sin_vals[2] + 0.05 * cos_vals[2] + 0.50 * cos_vals[1]
        self.J_toe[0, 1] = -0.26 * sin_vals[2] + 0.05 * cos_vals[2] + 0.50 * cos_vals[1]
        self.J_toe[0, 2] = -0.26 * sin_vals[2] + 0.05 * cos_vals[2]
        self.J_toe[2, 0] = 0.26 * cos_vals[2] + 0.05 * sin_vals[2] + 0.45 * sin_vals[0] + 0.50 * sin_vals[1]
        self.J_toe[2, 1] = 0.26 * cos_vals[2] + 0.05 * sin_vals[2] + 0.50 * sin_vals[1]
        self.J_toe[2, 2] = 0.26 * cos_vals[2] + 0.05 * sin_vals[2]

        # Jacobian for HT_heel
        self.J_heel[0, 0] = 0.45 * cos_vals[0] - 0.13 * sin_vals[2] + 0.05 * cos_vals[2] + 0.50 * cos_vals[1]
        self.J_heel[0, 1] = -0.13 * sin_vals[2] + 0.05 * cos_vals[2] + 0.50 * cos_vals[1]
        self.J_heel[0, 2] = -0.13 * sin_vals[2] + 0.05 * cos_vals[2]
        self.J_heel[2, 0] = -0.13 * cos_vals[2] + 0.05 * sin_vals[2] + 0.45 * sin_vals[0] + 0.50 * sin_vals[1]
        self.J_heel[2, 1] = -0.13 * cos_vals[2] + 0.05 * sin_vals[2] + 0.50 * sin_vals[1]
        self.J_heel[2, 2] = -0.13 * cos_vals[2] + 0.05 * sin_vals[2]

        # Update J_ankle

        self.J_ankle[0, 0] = 0.45 * cos_vals[0] + 0.5 * cos_vals[1]
        self.J_ankle[0, 1] = 0.5 * cos_vals[1]
        self.J_ankle[2, 0] = 0.45 * sin_vals[0] + 0.5 * sin_vals[1]
        self.J_ankle[2, 1] = 0.5 * sin_vals[1]

        # Update J_com1
        self.J_com1[0, 0] = 0.225 * cos_vals[0]
        self.J_com1[2, 0] = 0.225 * sin_vals[0]

        # Update J_com2
        self.J_com2[0, 0] = 0.45 * cos_vals[0] + 0.25 * cos_vals[1]
        self.J_com2[0, 1] = 0.25 * cos_vals[1]
        self.J_com2[2, 0] = 0.45 * sin_vals[0] + 0.25 * sin_vals[1]
        self.J_com2[2, 1] = 0.25 * sin_vals[1]

        # Update J_com3
        self.J_com3[0, 0] = 0.45 * cos_vals[0] - 0.065 * sin_vals[2] + 0.5 * cos_vals[1]
        self.J_com3[0, 1] = 0.5 * cos_vals[1] - 0.065 * sin_vals[2]
        self.J_com3[0, 2] = -0.065 * sin_vals[2]
        self.J_com3[2, 0] = 0.45 * sin_vals[0] + 0.065 * cos_vals[2] + 0.5 * sin_vals[1]
        self.J_com3[2, 1] = 0.5 * sin_vals[1] + 0.065 * cos_vals[2]
        self.J_com3[2, 2] = 0.065 * cos_vals[2]

    def update_com_pos(self):
        return (
            (
                self.HT_com0[0:3, 3] * self.m0
                + self.HT_com1[0:3, 3] * self.m1
                + self.HT_com2[0:3, 3] * self.m2
                + self.HT_com3[0:3, 3] * self.m3
            )
            / self.m
        ).reshape((3, 1))

    def update_inertia_tensor(self):
        Ib = np.zeros((3, 3), dtype=np.float64)
        for idx in range(len(self.masses)):
            rot = self.HT[idx][0:3, 0:3]
            trans = self.HT[idx][0:3, 3]
            Ib += rot @ self.Inertia[idx] @ rot.transpose() + (
                self.masses[idx] * np.dot(trans, trans) * np.eye(3) - np.outer(trans, trans)
            )
            # Ib += rot @ self.Inertia[idx] @ rot.transpose() + self.masses[idx] * sk_mtx @ sk_mtx.transpose()

        return Ib

    def _skew_mtx(self, vet):
        return np.array([[0, -vet[2], vet[1]], [vet[2], 0, -vet[0]], [-vet[1], vet[0], 0]])

    def com_jacobian(self):
        return (self.J_com1 * self.m1 + self.J_com2 * self.m2 + self.J_com3 * self.m3) / self.m


# model = ModelMatrices()
# model.update_robot_states(q=[0, 0, 0], dq=[0, 0, 0])
# model.update_homog_trans()
# print(model.update_inertia_tensor())
# print(model.HT[1])
