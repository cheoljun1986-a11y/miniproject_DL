import numpy as np
import gymnasium as gym
import os

# The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
# | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
# | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
# | 0   | x-coordinate of the torso                          | -Inf | Inf | rootz                            | slide | position (m)             |
# | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
# | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
# | 3   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
# | 4   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
# | 5   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
# | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
# | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
# | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
# | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
# | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
# | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
# | 12  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
# | 13  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
# | 14  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
# | 15  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
# | 16  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
# | 17  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

# The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
# | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
# |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
# | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
# | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
# | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
# | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
# | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10)
        
        super().__init__(env)
        
        ## change observation space according to the new observation
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, action, reward)
        custom_terminated = self.custom_terminated(terminated, obs)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_terminated(self, terminated, obs):
        if terminated:
            self.last_support_foot = None
            self.support_steps = 0

        return terminated
    
    def custom_truncated(self, truncated):
        if truncated:
            self.last_support_foot = None
            self.support_steps = 0

        return truncated

    def custom_observation(self, obs):
        # TODO: Implement your own observation
        return obs

    def custom_reward(self, obs, action, original_reward):
        # Solution #1: 전진, 에너지 절약, 상체의 안정적인 움직임을 유도한다.
        forward = obs[9]
        energy_reward = np.sum(np.square(action))
        stable_reward = (
            np.exp(-np.square(obs[2]))
            + np.exp(-np.square(obs[11]))
            + np.exp(-np.square(obs[10]))
        )

        # 전진 보상은 강화하고, 제자리에 서서 안정성 보상만 받는 행동은 억제한다.
        reward = 2.0 * forward - 1e-3 * energy_reward + 0.75 * stable_reward + 1.0

        # 지지 발을 번갈아 사용하고, 새 지지 발을 앞쪽에 착지하도록 유도한다.
        if not hasattr(self, "last_support_foot"):
            self.last_support_foot = None
            self.support_steps = 0

        mujoco_env = self.env.unwrapped
        right_foot_id = mujoco_env.model.geom("foot_geom").id
        left_foot_id = mujoco_env.model.geom("foot_left_geom").id

        # 평지뿐 아니라 범프도 발을 지지하는 지형으로 인식한다.
        terrain_ids = {
            mujoco_env.model.geom(i).id
            for i in range(mujoco_env.model.ngeom)
            if (
                mujoco_env.model.geom(i).name == "floor"
                or mujoco_env.model.geom(i).name.startswith("bump")
            )
        }

        right_foot_contact = False
        left_foot_contact = False
        for contact in mujoco_env.data.contact:
            contact_pair = {contact.geom1, contact.geom2}
            right_foot_contact |= (
                right_foot_id in contact_pair
                and bool(contact_pair & terrain_ids)
            )
            left_foot_contact |= (
                left_foot_id in contact_pair
                and bool(contact_pair & terrain_ids)
            )

        current_support_foot = None
        if right_foot_contact and not left_foot_contact:
            current_support_foot = "right"
        elif left_foot_contact and not right_foot_contact:
            current_support_foot = "left"

        right_knee_angle = obs[4]
        left_knee_angle = obs[7]
        swing_knee_target = -0.8
        stance_knee_target = -0.2
        knee_tolerance = 0.35

        if current_support_foot == "right":
            knee_gait_reward = (
                np.exp(-np.square((right_knee_angle - stance_knee_target) / knee_tolerance))
                + np.exp(-np.square((left_knee_angle - swing_knee_target) / knee_tolerance))
            )
        elif current_support_foot == "left":
            knee_gait_reward = (
                np.exp(-np.square((left_knee_angle - stance_knee_target) / knee_tolerance))
                + np.exp(-np.square((right_knee_angle - swing_knee_target) / knee_tolerance))
            )
        else:
            knee_gait_reward = 0.0

        if current_support_foot is not None:
            if current_support_foot == self.last_support_foot:
                self.support_steps += 1
            else:
                # 지지 발이 실제로 바뀌고 새 발이 앞에 착지할 때만 보상한다.
                if self.last_support_foot is not None:
                    right_foot_x = mujoco_env.data.geom(right_foot_id).xpos[0]
                    left_foot_x = mujoco_env.data.geom(left_foot_id).xpos[0]

                    if current_support_foot == "left":
                        step_length = left_foot_x - right_foot_x
                    else:
                        step_length = right_foot_x - left_foot_x

                    alternation_reward = np.clip(
                        step_length / 0.3,
                        0.0,
                        1.0,
                    )
                    reward += alternation_reward

                self.last_support_foot = current_support_foot
                self.support_steps = 1

        # 같은 발로 계속 버티는 행동은 보상을 중단하고 장시간 지속되면 감점한다.
        if 1 <= self.support_steps <= 20:
            reward += 0.25 * knee_gait_reward
        elif self.support_steps > 30:
            reward -= 0.2

        return reward

## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()
