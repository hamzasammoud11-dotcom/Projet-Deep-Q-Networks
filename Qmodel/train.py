import numpy as np
import random
import torch

from env_dataset import AntennaEnv
from dqn_agent import DQNAgent


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    # 1) Reproductibilité
    set_seed(0)

    # 2) Load dataset from professor
    Minput = np.load("../Dataset/Minput.npy")
    Moutput = np.load("../Dataset/Moutput.npy")

    # 3) Environment: all architectures, up to 10 steps per episode
    env = AntennaEnv(Minput, Moutput, max_steps=10)

    state_dim = 4
    action_dim = env.n_actions  # should be 1000

    # 4) DQN Agent with tuned epsilon decay
    # total_steps ≈ episodes * max_steps
    episodes = 2000
    max_steps_per_episode = env.max_steps
    total_steps_estimate = episodes * max_steps_per_episode  # 2000 * 10 = 20000

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=total_steps_estimate // 2,  # epsilon decays over ~half the training
        target_update=1000
    )

    reward_history = []

    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            # Choose action with epsilon-greedy policy
            a = agent.choose_action(s)

            # Interact with environment
            s2, r, done, _ = env.step(a)

            # Store transition and update DQN
            agent.buffer.push(s, a, r, s2, done)
            agent.update()

            s = s2
            ep_reward += r

            if done:
                break

        reward_history.append(ep_reward)

        # Logging
        if (ep + 1) % 50 == 0:
            last_rewards = reward_history[-50:]
            avg_last = sum(last_rewards) / len(last_rewards)
            print(f"Episode {ep+1}/{episodes}  "
                  f"Reward: {ep_reward:.2f}  "
                  f"Avg(50): {avg_last:.2f}  "
                  f"Eps: {agent.eps:.3f}")

    np.save("reward_history.npy", reward_history)


if __name__ == "__main__":
    train()
