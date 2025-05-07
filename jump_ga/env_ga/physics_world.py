# jump_ga/env_ga/physics_world.py

import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class PhysicsWorld:
    def __init__(self, num_robots, render=False):
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../jump_model/hopper.urdf"
        )
        self._time_step = 0.001
        self.render = render
        self.robot_ids = []
        self._last_frame_time = 0.0

        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._time_step)

        self._load_plane()
        self.load_robots(num_robots)

    def _load_plane(self):
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

    def load_robots(self, num_robots):
        spacing = 0.5
        self.robot_ids.clear()
        for i in range(num_robots):
            y_offset = i * spacing
            robot_id = p.loadURDF(self.model_path, basePosition=[0, y_offset, 0])
            self.robot_ids.append(robot_id)

    def reset_robot(self, index, model):
        q0 = model.randon_joint_pos()
        robot_id = self.robot_ids[index]
        for i in range(model.JOINT_MODEL_NUM):
            joint_id = model.JOINT_ST_LIST[i]
            p.resetJointState(robot_id, joint_id, q0[i, 0])
            p.setJointMotorControl2(robot_id, joint_id, p.VELOCITY_CONTROL, force=0)
        model.init_qr(q0[3:])  # pass joint angles only, skip base

    def step_all(self, robots):
        for i, robot in enumerate(robots):
            model = robot["model"]
            q_aux = np.zeros((model.JOINT_MODEL_NUM, 1), dtype=np.float64)
            dq_aux = np.zeros((model.JOINT_MODEL_NUM, 1), dtype=np.float64)
            f_cont = np.zeros((2, 1), dtype=np.float64)

            robot_id = self.robot_ids[i]
            for j in range(model.JOINT_MODEL_NUM):
                joint_id = model.JOINT_ST_LIST[j]
                q, dq, _, _ = p.getJointState(robot_id, joint_id)
                q_aux[j, 0] = q
                dq_aux[j, 0] = dq

            contact_ids = model.CONTACT_JOINTS
            for j, link_id in enumerate(contact_ids):
                contact = p.getContactPoints(bodyA=robot_id, linkIndexA=link_id)
                f_cont[j, 0] = 1 if contact else 0

            model.update_robot_states(q_aux, dq_aux, f_cont)
            tau = model.command_torque()
            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=model.AC_JOINT_LIST,
                controlMode=p.TORQUE_CONTROL,
                forces=tau.flatten().tolist(),
            )

        p.stepSimulation()

        if self.render:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            sleep_time = self._time_step - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)
