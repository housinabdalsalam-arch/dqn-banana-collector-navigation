# Banana Collector (DQN) — Udacity DRL Project 1

This repository contains a **PyTorch Deep Q-Network (DQN)** solution for Udacity’s **Navigation / Banana Collector** environment. The agent learns to collect **yellow bananas (+1)** while avoiding **blue bananas (-1)**.

## Demo

![Trained agent demo](assets/banana_agent.gif)

## Environment

- **State space:** 37 continuous values (velocity + ray-based perception)
- **Action space:** 4 discrete actions — `0` forward · `1` backward · `2` turn left · `3` turn right
- **Solved when:** average score **>= 13** over **100** consecutive episodes

## Repository contents

- `src/model.py` — Q-network 
- `src/replay_buffer.py` — replay buffer
- `src/dqn_agent.py` — DQN agent (epsilon-greedy policy, learning update, target soft update)
- `src/train.py` — training script (saves `checkpoint.pth` and `scores.png`)
- `src/evaluate.py` — evaluation script (runs a trained agent from `checkpoint.pth`)
- `scores.png` — training curve
- `checkpoint.pth` — trained weights

## Setup

### 1) Download the Unity environment

Download and unzip the environment into the **repository root**.

- **Windows (64-bit):** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
- **Linux:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- **Mac:** [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Expected path after unzipping (Windows):
```text
Banana_Windows_x86_64/Banana.exe
```

### 2) Install dependencies

Create/activate a virtual environment, then install:

```bash
pip install numpy matplotlib torch
pip install protobuf==3.20.3 grpcio==1.26.0
pip install unityagents==0.4.0 --no-deps
```

## Train

From the repository root:

```bash
python src/train.py
```

Training stops once solved and writes:
- `checkpoint.pth`
- `scores.png`

## Evaluate

```bash
python src/evaluate.py
```

Loads `checkpoint.pth` and runs a few episodes with a greedy policy (`eps=0`).

## Results

- Solved in **583 episodes**
- Average score (last 100 episodes): **13.02**
- Evaluation mean score (10 episodes): **15.10**

