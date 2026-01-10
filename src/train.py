from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from dqn_agent import Agent, DQNConfig


def train(env_path: Path,
          n_episodes: int = 2000,
          max_t: int = 1000,
          eps_start: float = 1.0,
          eps_end: float = 0.01,
          eps_decay: float = 0.995,
          solved_score: float = 13.0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = UnityEnvironment(file_name=str(env_path))
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Probe sizes
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    print("Brain:", brain_name)
    print("State size:", state_size)
    print("Action size:", action_size)

    agent = Agent(state_size, action_size, DQNConfig(seed=0), device=device)

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    t0 = time.time()

    try:
        for i_episode in range(1, n_episodes + 1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0.0

            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = float(env_info.rewards[0])
                done = bool(env_info.local_done[0])

                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)
            scores.append(score)

            eps = max(eps_end, eps_decay * eps)

            avg = np.mean(scores_window)
            print(f"\rEpisode {i_episode}\tAverage(100): {avg:.2f}\tEps: {eps:.3f}", end="")

            if i_episode % 100 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"\rEpisode {i_episode}\tAverage(100): {avg:.2f}\tElapsed: {elapsed:.1f} min")

            if avg >= solved_score:
                print(f"\nSolved in {i_episode} episodes! Average(100): {avg:.2f}")
                torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
                break

    finally:
        env.close()

    # Plot
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("DQN Training")
    plt.savefig("scores.png", dpi=150, bbox_inches="tight")
    plt.show()

    return scores


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    env_path = root / "Banana_Windows_x86_64" / "Banana.exe"
    train(env_path)
