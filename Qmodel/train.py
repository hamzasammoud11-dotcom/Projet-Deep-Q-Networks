import numpy as np
from env_dataset import AntennaEnv
from dqn_agent import DQNAgent

def train():
    Minput = np.load("../Dataset/Minput.npy")
    Moutput = np.load("../Dataset/Moutput.npy")

    env = AntennaEnv(Minput, Moutput, max_steps=10)

    state_dim = 4
    action_dim = env.n_actions

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        batch_size=64
    )

    episodes = 300
    reward_history = []

    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0

        for t in range(10):
            a = agent.choose_action(s)
            s2, r, done, _ = env.step(a)
            agent.buffer.push(s, a, r, s2, done)
            agent.update()

            s = s2
            ep_reward += r
            if done:
                break

        reward_history.append(ep_reward)

        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}  Reward: {ep_reward:.2f}")

    np.save("reward_history.npy", reward_history)

if __name__ == "__main__":
    train()
