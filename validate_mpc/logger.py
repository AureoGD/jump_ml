import os
import csv

class MPCLogger:
    def __init__(self, log_dir, filename="mpc_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.data = []

    def log(self, step, robot_states):
        # You can extract any fields you want here
        row = {
            "step": step,
            "q": robot_states.q.tolist(),
            "dq": robot_states.dq.tolist(),
            "qr": robot_states.qr.tolist(),
            "tau": robot_states.tau.tolist(),
            "r_pos": robot_states.r_pos.tolist(),
            "r_vel": robot_states.r_vel.tolist(),
            "th": robot_states.th.tolist(),
            "dth": robot_states.dth.tolist(),
        }
        self.data.append(row)

    def save(self):
        if not self.data:
            return

        keys = self.data[0].keys()
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)
