"""Reproducible DQN experiment for Gymnasium Acrobot-v1."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENV_ID = "Acrobot-v1"
VELOCITY_LIMITS = np.array([4 * np.pi, 9 * np.pi], dtype=np.float32)


@dataclass(frozen=True)
class Config:
    episodes: int = 600
    max_steps: int = 500
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    train_every: int = 4
    target_every: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 80_000
    eval_every: int = 50
    eval_episodes: int = 20
    final_eval_episodes: int = 50
    hidden1: int = 128
    hidden2: int = 64


def features(observation: np.ndarray) -> np.ndarray:
    """Keep trigonometric coordinates and scale angular velocities to [-1, 1]."""
    x = np.asarray(observation, dtype=np.float32).copy()
    if x.shape != (6,):
        raise ValueError(f"Expected six observations, got {x.shape}")
    x[4:] /= VELOCITY_LIMITS
    return x


def tip_height(observation: np.ndarray) -> float:
    """Tip height relative to the fixed joint, computed from observed angles."""
    c1, s1, c2, s2 = np.asarray(observation)[:4]
    return float(-c1 - (c1 * c2 - s1 * s2))


class QNetwork(nn.Module):
    def __init__(self, hidden1: int = 128, hidden2: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, 6), dtype=np.float32)
        self.next_states = np.empty((capacity, 6), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions for a batch")
        indexes = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_states[indexes],
            self.terminated[indexes],
        )


def epsilon_at(step: int, config: Config) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def greedy_action(model: QNetwork, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        values = model(torch.as_tensor(state, device=device).unsqueeze(0))
    return int(values.argmax(dim=1).item())


def update_dqn(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    config: Config,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, terminated = buffer.sample(
        config.batch_size
    )
    s = torch.as_tensor(states, device=device)
    a = torch.as_tensor(actions, device=device).unsqueeze(1)
    r = torch.as_tensor(rewards, device=device)
    next_s = torch.as_tensor(next_states, device=device)
    terminal = torch.as_tensor(terminated, device=device)

    predicted = online(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        target_value = r + config.gamma * (1 - terminal) * target(next_s).max(1).values
    loss = F.smooth_l1_loss(predicted, target_value)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def evaluate(
    model: QNetwork | None,
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    env = gym.make(ENV_ID)
    records = []
    try:
        for episode_seed in seeds:
            observation, _ = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed)
            total_reward = 0.0
            maximum_height = tip_height(observation)
            for steps in range(1, 501):
                action = (
                    int(rng.integers(3))
                    if model is None
                    else greedy_action(model, features(observation), device)
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                maximum_height = max(maximum_height, tip_height(observation))
                if terminated or truncated:
                    break
            records.append(
                {
                    "seed": episode_seed,
                    "reward": total_reward,
                    "steps": steps,
                    "success": bool(terminated),
                    "maximum_height": maximum_height,
                }
            )
    finally:
        env.close()
    return records


def summarize(records: list[dict]) -> dict[str, float]:
    rewards = np.array([row["reward"] for row in records], dtype=float)
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "median_reward": float(np.median(rewards)),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_steps": float(np.mean([row["steps"] for row in records])),
        "mean_maximum_height": float(
            np.mean([row["maximum_height"] for row in records])
        ),
    }


def save_rows(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def train(seed: int, config: Config, output: Path, device: torch.device) -> dict:
    seed_everything(seed)
    torch.set_num_threads(1)
    env = gym.make(ENV_ID)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    online = QNetwork(config.hidden1, config.hidden2).to(device)
    target = QNetwork(config.hidden1, config.hidden2).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.replay_capacity, seed)
    history = []
    evaluations = []
    global_step = 0
    best_reward = float("-inf")
    output.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(1, config.episodes + 1):
            observation, _ = env.reset(seed=seed if episode == 1 else None)
            state = features(observation)
            total_reward = 0.0
            maximum_height = tip_height(observation)
            episode_losses = []
            for steps in range(1, config.max_steps + 1):
                epsilon = epsilon_at(global_step, config)
                action = (
                    int(rng.integers(3))
                    if rng.random() < epsilon
                    else greedy_action(online, state, device)
                )
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = features(next_observation)
                buffer.add(state, action, reward, next_state, terminated)
                total_reward += reward
                maximum_height = max(maximum_height, tip_height(next_observation))
                global_step += 1
                if buffer.size >= max(config.batch_size, config.warmup_steps) and global_step % config.train_every == 0:
                    episode_losses.append(
                        update_dqn(online, target, optimizer, buffer, config, device)
                    )
                if global_step % config.target_every == 0:
                    target.load_state_dict(online.state_dict())
                state = next_state
                if terminated or truncated:
                    break
            history.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "global_step": global_step,
                    "reward": total_reward,
                    "steps": steps,
                    "success": bool(terminated),
                    "epsilon": epsilon_at(global_step, config),
                    "mean_loss": float(np.mean(episode_losses)) if episode_losses else float("nan"),
                    "maximum_height": maximum_height,
                }
            )
            if episode % config.eval_every == 0 or episode == config.episodes:
                eval_seeds = list(range(100_000, 100_000 + config.eval_episodes))
                metrics = summarize(evaluate(online, eval_seeds, device))
                evaluations.append({"seed": seed, "episode": episode, **metrics})
                print(
                    f"seed={seed} episode={episode}/{config.episodes} "
                    f"steps={global_step} eval_reward={metrics['mean_reward']:.1f} "
                    f"success={metrics['success_rate']:.0%}",
                    flush=True,
                )
                if metrics["mean_reward"] > best_reward:
                    best_reward = metrics["mean_reward"]
                    torch.save(online.state_dict(), output / "best.pt")
    finally:
        env.close()
        save_rows(output / "training.csv", history)
        save_rows(output / "validation.csv", evaluations)
    torch.save(online.state_dict(), output / "final.pt")
    return {"seed": seed, "best_validation_reward": best_reward, "steps": global_step}


def plot_learning(paths: list[Path], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for path in paths:
        rows = list(csv.DictReader((path / "training.csv").open(encoding="utf-8")))
        episode = np.array([int(row["episode"]) for row in rows])
        reward = np.array([float(row["reward"]) for row in rows])
        height = np.array([float(row["maximum_height"]) for row in rows])
        window = min(25, len(reward))
        if window:
            averaged = np.convolve(reward, np.ones(window) / window, mode="valid")
            axes[0].plot(episode[window - 1 :], averaged, label=path.name)
            height_average = np.convolve(height, np.ones(window) / window, mode="valid")
            axes[1].plot(episode[window - 1 :], height_average, label=path.name)
    axes[0].axhline(-100, color="#b24a30", linestyle="--", label="referencia -100")
    axes[0].set(title="Recompensa de entrenamiento", xlabel="Episodio", ylabel="Media movil (25 episodios)")
    axes[1].axhline(1, color="#b24a30", linestyle="--", label="altura objetivo")
    axes[1].set(title="Altura maxima observada", xlabel="Episodio", ylabel="Media movil (25 episodios)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_validation(paths: list[Path], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for path in paths:
        rows = list(csv.DictReader((path / "validation.csv").open(encoding="utf-8")))
        episode = [int(row["episode"]) for row in rows]
        reward = [float(row["mean_reward"]) for row in rows]
        success = [float(row["success_rate"]) for row in rows]
        axes[0].plot(episode, reward, marker="o", markersize=3, label=path.name)
        axes[1].plot(episode, success, marker="o", markersize=3, label=path.name)
    axes[0].axhline(-100, color="#b24a30", linestyle="--", label="referencia -100")
    axes[0].set(title="Recompensa de validacion greedy", ylabel="Media de 20 episodios")
    axes[1].set(title="Exito de validacion greedy", ylabel="Proporcion de 20 episodios", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.set_xlabel("Episodio de entrenamiento")
        axis.grid(alpha=0.2)
        axis.legend()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run_experiment(config: Config, seeds: list[int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (output / "config.json").write_text(
        json.dumps({"config": asdict(config), "seeds": seeds, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    baseline_seeds = list(range(200_000, 200_000 + config.final_eval_episodes))
    baseline = evaluate(None, baseline_seeds, device)
    save_rows(output / "random_baseline.csv", baseline)
    all_results = []
    seed_paths = []
    for seed in seeds:
        seed_dir = output / f"seed_{seed}"
        train(seed, config, seed_dir, device)
        model = QNetwork(config.hidden1, config.hidden2).to(device)
        model.load_state_dict(torch.load(seed_dir / "best.pt", map_location=device, weights_only=True))
        model.eval()
        final = evaluate(model, baseline_seeds, device)
        save_rows(seed_dir / "test.csv", final)
        all_results.append({"seed": seed, **summarize(final)})
        seed_paths.append(seed_dir)
    plot_learning(seed_paths, output / "learning_curve.png")
    plot_validation(seed_paths, output / "validation_curve.png")
    summary = {"random_baseline": summarize(baseline), "dqn_by_seed": all_results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Acrobot DQN")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--eval-every", type=int, default=50)
    arguments = parser.parse_args()
    config = Config(episodes=arguments.episodes, eval_every=arguments.eval_every)
    run_experiment(config, arguments.seeds, arguments.output)


if __name__ == "__main__":
    main()
