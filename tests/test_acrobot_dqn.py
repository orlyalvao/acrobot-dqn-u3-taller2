"""Focused checks for observations, replay, and terminal Bellman updates."""

import unittest

import gymnasium as gym
import numpy as np
import torch

from acrobot_dqn import Config, QNetwork, ReplayBuffer, epsilon_at, features, tip_height, update_dqn


class AcrobotDQNTests(unittest.TestCase):
    def test_gymnasium_contract(self) -> None:
        env = gym.make("Acrobot-v1")
        try:
            observation, _ = env.reset(seed=2)
            self.assertEqual(observation.shape, (6,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertEqual(env.action_space.n, 3)
            self.assertEqual(env.spec.max_episode_steps, 500)
            _, reward, terminated, truncated, _ = env.step(1)
            self.assertIn(reward, (-1.0, 0.0))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
        finally:
            env.close()

    def test_features_scale_only_velocities(self) -> None:
        observation = np.array([1, 0, 0, 1, 4 * np.pi, -9 * np.pi], dtype=np.float32)
        transformed = features(observation)
        np.testing.assert_allclose(transformed, [1, 0, 0, 1, 1, -1], atol=1e-6)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertEqual(observation[-1], np.float32(-9 * np.pi))

    def test_tip_height_from_trigonometry(self) -> None:
        self.assertAlmostEqual(tip_height(np.array([1, 0, 1, 0, 0, 0])), -2)
        self.assertAlmostEqual(tip_height(np.array([-1, 0, 1, 0, 0, 0])), 2)

    def test_epsilon_schedule(self) -> None:
        config = Config()
        self.assertEqual(epsilon_at(0, config), 1.0)
        self.assertAlmostEqual(epsilon_at(80_000, config), 0.05)
        self.assertAlmostEqual(epsilon_at(90_000, config), 0.05)

    def test_replay_capacity_and_shapes(self) -> None:
        replay = ReplayBuffer(3, seed=7)
        for i in range(5):
            replay.add(np.full(6, i), i % 3, -1, np.full(6, i + 1), False)
        self.assertEqual(replay.size, 3)
        states, actions, rewards, next_states, terminal = replay.sample(2)
        self.assertEqual(states.shape, (2, 6))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(next_states.shape, (2, 6))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(terminal.shape, (2,))

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        config = Config(batch_size=1, warmup_steps=1)
        online = QNetwork()
        target = QNetwork()
        with torch.no_grad():
            for parameter in online.parameters():
                parameter.zero_()
            for parameter in target.parameters():
                parameter.zero_()
            target.layers[-1].bias.fill_(100)
        replay = ReplayBuffer(2, seed=1)
        replay.add(np.zeros(6), 0, -1, np.zeros(6), True)
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        loss = update_dqn(online, target, optimizer, replay, config, torch.device("cpu"))
        self.assertAlmostEqual(loss, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
