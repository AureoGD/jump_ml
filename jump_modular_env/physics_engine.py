import pybullet as p
import pybullet_data
import numpy as np
import time


class PhysicsEngine:
    def __init__(self, robot_model, render=False):
        self.robot_model = robot_model
        self._is_render = render
        self._time_step = robot_model.sim_dt

        self.physics_client = p.connect(p.GUI if self._is_render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.model = None
        self.num_joints = 0
        self.q_aux = np.zeros((robot_model.JOINT_MODEL_NUM, 1), dtype=np.float64)
        self.dq_aux = np.zeros((robot_model.JOINT_MODEL_NUM, 1), dtype=np.float64)
        self.f_cont = np.zeros((len(robot_model.CONTACT_JOINTS), 1), dtype=np.float64)

        self._last_frame_time = 0.0

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._time_step)
        p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])

        self.model = p.loadURDF(self.robot_model.model_path, [0, 0, 0], [0, 0, 0, 1])
        self.num_joints = p.getNumJoints(self.model)
        p.changeDynamics(self.model, self.num_joints - 1, lateralFriction=1.5)

    def initialize_joint_states(self, q):
        for idx in range(len(q)):
            p.resetJointState(self.model, self.robot_model.JOINT_ST_LIST[idx], q[idx, 0])
            p.setJointMotorControl2(self.model, self.robot_model.JOINT_ST_LIST[idx], p.VELOCITY_CONTROL, force=0)

    def get_joint_state(self):
        for idx in range(self.robot_model.JOINT_MODEL_NUM):
            self.q_aux[idx], self.dq_aux[idx], _, _ = p.getJointState(self.model, self.robot_model.JOINT_ST_LIST[idx])

        for idx in range(len(self.robot_model.CONTACT_JOINTS)):
            contacts = p.getContactPoints(bodyA=self.model, linkIndexA=self.robot_model.CONTACT_JOINTS[idx])
            self.f_cont[idx, 0] = 1.0 if len(contacts) > 0 else 0.0

        return self.q_aux.copy(), self.dq_aux.copy(), self.f_cont.copy()

    def apply_torque(self, tau):
        p.setJointMotorControlArray(
            self.model,
            self.robot_model.AC_JOINT_LIST,
            p.TORQUE_CONTROL,
            forces=tau,
        )

    def step(self):
        p.stepSimulation()
        if self._is_render:
            elapsed = time.time() - self._last_frame_time
            sleep_time = self._time_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_frame_time = time.time()
