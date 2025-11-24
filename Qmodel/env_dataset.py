import numpy as np

class AntennaEnv:
    def __init__(self, Minput, Moutput, max_steps=10):
        self.Minput = Minput.T   # shape (N,4)
        self.Moutput = Moutput.T # shape (N,5)
        self.n_actions = self.Minput.shape[0]
        self.max_steps = max_steps
        self.t = 0

        # Normalisation
        self.min_vals = self.Minput.min(axis=0)
        self.max_vals = self.Minput.max(axis=0)

    def normalize(self, x):
        return (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)

    def reset(self):
        self.t = 0
        idx = np.random.randint(0, self.n_actions)
        self.state = self.normalize(self.Minput[idx])
        return self.state

    def step(self, action):
        self.t += 1

        next_state = self.normalize(self.Minput[action])
        self.state = next_state

        # Définition de la récompense
        MainLobe, SSL, HPBW, Theta0 = self.Minput[action]
        reward = MainLobe - abs(SSL) - HPBW

        done = self.t >= self.max_steps

        return next_state, reward, done, {}
