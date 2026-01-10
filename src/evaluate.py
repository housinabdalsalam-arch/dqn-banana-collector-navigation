from pathlib import Path
import numpy as np
import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent, DQNConfig


def main():
    device = torch.device("cpu")  # you trained on CPU, keep it consistent
    root = Path(__file__).resolve().parents[1]
    env_path = root / "Banana_Windows_x86_64" / "Banana.exe"
    weights_path = root / "checkpoint.pth"

    env = UnityEnvironment(file_name=str(env_path))
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size

    agent = Agent(state_size, action_size, DQNConfig(seed=0), device=device)
    agent.qnetwork_local.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    scores = []
    for ep in range(5):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0.0

        while True:
            action = agent.act(state, eps=0.0)  # greedy
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            score += float(env_info.rewards[0])
            if bool(env_info.local_done[0]):
                break

        scores.append(score)
        print(f"Episode {ep+1}: score = {score:.2f}")

    env.close()
    print(f"Mean score over {len(scores)} episodes: {np.mean(scores):.2f}")


if __name__ == "__main__":
    main()
