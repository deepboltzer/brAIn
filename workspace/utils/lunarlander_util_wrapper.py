import numpy as np
import math
import gym
from gym.envs.box2d import lunar_lander as ll
from gym.utils import EzPickle


class LunarLanderUtilWrapper(gym.Wrapper):
    """
    Wrapper that supplies a utility function to evaluate actions without changing the current state.
    :param env: instance of the LunarLander-v2 environment
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def __reduce__(self):
        """"""
        return (self.__class__, (self.screen, self.isopen, self.world, self.moon, self.lander, self.particles,))
    
    def test_action(self, action):
        """Performs a step without applying changes to the state."""
        
        print(f"Lander before action test: {self.lander}")
        # # Save state
        # save_lander_pos = self.lander.position
        # save_lander_angle = self.lander.angle
        # save_lander_linVel = self.lander.linearVelocity
        # save_lander_angVel = self.lander.angularVelocity

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / ll.SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = (
                tip[0] * (4 / ll.SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            )  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / ll.SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            # No need for particles
            # p = self._create_particle(
            #     3.5,  # 3.5 is here to make particle speed adequate
            #     impulse_pos[0],
            #     impulse_pos[1],
            #     m_power,
            # )  # particles are just a decoration
            # p.ApplyLinearImpulse(
            #     (ox * ll.MAIN_ENGINE_POWER * m_power, oy * ll.MAIN_ENGINE_POWER * m_power),
            #     impulse_pos,
            #     True,
            # )
            self.lander.ApplyLinearImpulse(
                (-ox * ll.MAIN_ENGINE_POWER * m_power, -oy * ll.MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * ll.SIDE_ENGINE_AWAY / ll.SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * ll.SIDE_ENGINE_AWAY / ll.SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / ll.SCALE,
                self.lander.position[1] + oy + tip[1] * ll.SIDE_ENGINE_HEIGHT / ll.SCALE,
            )
            # No need for particles
            # p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            # p.ApplyLinearImpulse(
            #     (ox * ll.SIDE_ENGINE_POWER * s_power, oy * ll.SIDE_ENGINE_POWER * s_power),
            #     impulse_pos,
            #     True,
            # )
            self.lander.ApplyLinearImpulse(
                (-ox * ll.SIDE_ENGINE_POWER * s_power, -oy * ll.SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        # Fortunately it doesn't seem that this affects the reward
        self.world.Step(1.0 / ll.FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - ll.VIEWPORT_W / ll.SCALE / 2) / (ll.VIEWPORT_W / ll.SCALE / 2),
            (pos.y - (self.helipad_y + ll.LEG_DOWN / ll.SCALE)) / (ll.VIEWPORT_H / ll.SCALE / 2),
            vel.x * (ll.VIEWPORT_W / ll.SCALE / 2) / ll.FPS,
            vel.y * (ll.VIEWPORT_H / ll.SCALE / 2) / ll.FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / ll.FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        
        # # revert changes
        # self.lander.position = save_lander_pos
        # self.lander.angle = save_lander_angle
        # self.lander.linearVelocity = save_lander_linVel
        # self.lander.angularVelocity = save_lander_angVel

        
        print(f"Lander after action test: {self.lander}")

        return np.array(state, dtype=np.float32), reward, done, {}