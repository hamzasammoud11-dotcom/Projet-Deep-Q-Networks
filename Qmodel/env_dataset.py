import numpy as np

class AntennaEnv:
    """
    RL environment built on top of the dataset outputs (Minput, Moutput).

    - Each ACTION corresponds to choosing one antenna architecture (one column of Moutput).
    - The STATE is the normalized RF performance vector [MainLobe, SSL, HPBW, Theta0].
    - The REWARD is a normalized scalar combining MainLobe, SSL and HPBW.
    """

    def __init__(self, Minput, Moutput, max_steps=10, max_actions=100, seed=None):
        # Optional seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Minput: shape (4, N)
        # Moutput: shape (5, N)
        self.Minput_full = Minput    # keep original
        self.Moutput_full = Moutput

        num_samples = Minput.shape[1]

        # Select a subset of actions if max_actions < N
        if max_actions is not None and max_actions < num_samples:
            self.indices = np.random.choice(num_samples, size=max_actions, replace=False)
        else:
            self.indices = np.arange(num_samples)

        # Reduced matrices, one row per action
        self.Minput = Minput[:, self.indices].T   # shape (n_actions, 4)
        self.Moutput = Moutput[:, self.indices].T # shape (n_actions, 5)

        self.n_actions = self.Minput.shape[0]
        self.max_steps = max_steps
        self.t = 0

        # --- Normalization for state (features) ---
        self.state_min = self.Minput.min(axis=0)
        self.state_max = self.Minput.max(axis=0)

        # --- Normalization for reward components ---
        # MainLobe, SSL, HPBW are the first three components
        self.ml_min, self.ml_max = self.Minput[:, 0].min(), self.Minput[:, 0].max()
        self.ssl_min, self.ssl_max = self.Minput[:, 1].min(), self.Minput[:, 1].max()
        self.hpbw_min, self.hpbw_max = self.Minput[:, 2].min(), self.Minput[:, 2].max()

        self.state = None

    def _normalize_state(self, x):
        return (x - self.state_min) / (self.state_max - self.state_min + 1e-8)

    def _compute_reward(self, raw_features):
        """
        raw_features = [MainLobe, SSL, HPBW, Theta0] (unnormalized)
        We normalize each component and then build a reward.
        """
        MainLobe, SSL, HPBW, Theta0 = raw_features

        ml_norm   = (MainLobe - self.ml_min) / (self.ml_max - self.ml_min + 1e-8)
        ssl_norm  = (SSL      - self.ssl_min) / (self.ssl_max - self.ssl_min + 1e-8)
        hpbw_norm = (HPBW     - self.hpbw_min) / (self.hpbw_max - self.hpbw_min + 1e-8)

        # We want:
        # - high MainLobe  -> good
        # - low SSL        -> good  (so we subtract ssl_norm)
        # - low HPBW       -> good  (so we subtract hpbw_norm)
        reward = ml_norm - ssl_norm - hpbw_norm

        return float(reward)

    def reset(self):
        """
        Start a new episode by picking a random architecture and returning its normalized RF performance.
        """
        self.t = 0
        idx = np.random.randint(0, self.n_actions)
        raw_state = self.Minput[idx]         # [MainLobe, SSL, HPBW, Theta0]
        self.state = self._normalize_state(raw_state)
        return self.state

    def step(self, action):
        """
        Apply the chosen action = select a given architecture index.
        Return:
        - next_state: normalized RF performance
        - reward: scalar, normalized
        - done: whether the episode is over
        - info: empty dict (placeholder)
        """
        self.t += 1

        # Safety: clip action index
        action = int(action) % self.n_actions

        raw_features = self.Minput[action]   # unnormalized RF features
        next_state = self._normalize_state(raw_features)
        self.state = next_state

        reward = self._compute_reward(raw_features)

        done = self.t >= self.max_steps

        return next_state, reward, done, {}
