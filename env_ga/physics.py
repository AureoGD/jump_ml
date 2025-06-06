# env_ga/physics_world.py
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import Dict, Any


# --- Disturbance Class ---
# This can be moved to its own file (e.g., utils/disturbance.py) and imported.
class Disturbance:

    def __init__(self, model_id, config: Dict[str, Any]):
        """
        Applies random external forces to a model to test robustness.

        Args:
            model_id (int): The unique ID of the robot model in PyBullet.
            config (Dict): A dictionary containing disturbance parameters.
                Expected keys: 'link_index', 'force_magnitude', 'duration', 'chance', 'start_step'.
        """
        self.model_id = model_id
        self.link_index = config.get("link_index", 2)  # Default link (e.g., torso)
        self.force_magnitude = config.get("force_magnitude", 100.0)
        self.duration = config.get("duration", 10)  # Number of simulation steps to apply force
        self.chance = config.get("chance", 0.05)  # Chance to apply disturbance in a given episode
        self.start_step = config.get("start_step", 600)  # Step after which disturbances can occur

        self.remaining_duration = 0
        self.force_to_apply = np.zeros(3)
        self.is_active_in_episode = False

    def reset(self):
        """Resets the disturbance state for a new episode."""
        self.remaining_duration = 0
        self.force_to_apply = np.zeros(3)
        # Decide if a disturbance will happen in this episode
        self.is_active_in_episode = (np.random.rand() < self.chance)
        if self.is_active_in_episode:
            # Pre-calculate the force vector for this episode
            fx = np.random.uniform(-1, 1)
            fz = np.random.uniform(-1, 1)
            force_direction = np.array([fx, 0.0, fz])
            norm = np.linalg.norm(force_direction)
            if norm > 0:
                self.force_to_apply = (force_direction / norm) * self.force_magnitude
            print(f"[Disturbance] Armed for this episode with potential force: {self.force_to_apply}")

    def apply(self, current_step: int, client_id: int):
        """Applies the disturbance force if conditions are met."""
        if not self.is_active_in_episode:
            return

        # Trigger the disturbance if it hasn't started yet and we are past the start step
        if self.remaining_duration == 0 and current_step >= self.start_step:
            self.remaining_duration = self.duration
            print(
                f"[Disturbance] Applying force {self.force_to_apply} at step {current_step} for {self.duration} steps.")

        # Apply force if it's currently active
        if self.remaining_duration > 0:
            p.applyExternalForce(
                objectUniqueId=self.model_id,
                linkIndex=self.link_index,
                forceObj=self.force_to_apply.tolist(),
                posObj=[0, 0, 0],  # Applying at the link's center of mass
                flags=p.WORLD_FRAME,
                physicsClientId=client_id)
            self.remaining_duration -= 1


# --- Main PhysicsWorld Class ---


class PhysicsWorld:
    """
    Manages a single PyBullet physics simulation instance for one robot.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PyBullet client and loads the environment.

        Args:
            config (Dict): A configuration dictionary. Expected keys:
                - 'render' (bool)
                - 'sim_dt' (float)
                - 'robot_urdf_path' (str)
                - 'ground_urdf_path' (str, optional)
                - 'apply_disturbance' (bool, optional)
                - 'disturbance_config' (Dict, optional)
        """
        self.render = config.get("render", False)
        self.time_step = config.get("sim_dt", 0.001)
        self.robot_urdf_path = config.get("robot_urdf_path",
                                          os.path.join(os.path.dirname(__file__), "../jump_model/hopper.urdf"))

        self.client_id = p.connect(p.GUI if self.render else p.DIRECT)
        if self.client_id < 0:
            raise ConnectionError("Could not connect to PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id)
            self._last_frame_time = time.time()

        # Load a custom ground or the default plane
        ground_urdf_path = config.get("ground_urdf_path", "plane.urdf")
        self.plane_id = p.loadURDF(ground_urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client_id)
        print(f"Loaded ground from: {ground_urdf_path}")

        start_pos = config.get("start_pos", [0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf_path, start_pos, physicsClientId=self.client_id)

        # Initialize disturbance handler if configured
        self.apply_disturbance = config.get("apply_disturbance", False)
        self.disturbance = None
        if self.apply_disturbance:
            disturbance_config = config.get("disturbance_config", {})
            self.disturbance = Disturbance(self.robot_id, disturbance_config)
            print("Disturbance handler initialized.")

    def reset_robot(self, robot_model):
        """Resets the robot to an initial state in the simulation."""
        # ... (reset logic as before) ...
        # start_pos = self.config.get("start_pos", [0, 0, 1])
        # start_pos = [0, 0, 1]
        # p.resetBasePositionAndOrientation(self.robot_id, start_pos, [0, 0, 0, 1], physicsClientId=self.client_id)
        initial_joint_config = robot_model.randon_joint_pos()
        for i in range(robot_model.JOINT_MODEL_NUM):
            joint_id = robot_model.JOINT_ST_LIST[i]
            p.setJointMotorControl2(self.robot_id,
                                    joint_id,
                                    p.VELOCITY_CONTROL,
                                    force=0,
                                    physicsClientId=self.client_id)
            p.resetJointState(self.robot_id,
                              joint_id,
                              initial_joint_config[i, 0],
                              targetVelocity=0.0,
                              physicsClientId=self.client_id)
        robot_model.init_qr(initial_joint_config[3:])

        # Reset the disturbance handler for the new episode
        if self.disturbance:
            self.disturbance.reset()
        self._update_model_from_sim(robot_model)

    def step(self, robot_model, current_episode_step: int):
        """
        Executes one full simulation step.
        """
        # 1. Read current state from PyBullet and update the HopperModel
        self._update_model_from_sim(robot_model)

        # 2. Get torque command from the HopperModel
        torques = robot_model.command_torque()

        # 3. Apply torques to the robot's joints
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=robot_model.AC_JOINT_LIST,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=torques.flatten().tolist(),
                                    physicsClientId=self.client_id)

        # 4. Apply external disturbance force if active
        if self.disturbance:
            self.disturbance.apply(current_episode_step, self.client_id)

        # 5. Advance the physics simulation
        p.stepSimulation(physicsClientId=self.client_id)

        # 6. Real-time rendering sleep logic
        if self.render:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            sleep_time = self.time_step - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_model_from_sim(self, robot_model):
        """Helper to read physical state from PyBullet and update the HopperModel."""
        q_aux = np.zeros((robot_model.JOINT_MODEL_NUM, 1), dtype=np.float32)
        dq_aux = np.zeros((robot_model.JOINT_MODEL_NUM, 1), dtype=np.float32)

        for i in range(robot_model.JOINT_MODEL_NUM):
            joint_id = robot_model.JOINT_ST_LIST[i]
            q, dq, _, _ = p.getJointState(self.robot_id, joint_id, physicsClientId=self.client_id)
            q_aux[i, 0] = q
            dq_aux[i, 0] = dq

        f_cont = np.zeros((2, 1), dtype=np.float32)
        for j, link_id in enumerate(robot_model.CONTACT_JOINTS):
            contact = p.getContactPoints(bodyA=self.robot_id, linkIndexA=link_id, physicsClientId=self.client_id)
            f_cont[j, 0] = 1 if contact else 0

        robot_model.update_robot_states(q_aux, dq_aux, f_cont)

    def disconnect(self):
        """Disconnects from the PyBullet server."""
        if p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
