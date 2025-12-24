"""
env_antenna.py - Proper RL Environment for Antenna Optimization
================================================================
This environment implements a CORRECT MDP formulation for DQN-based 
antenna array optimization.

THE KEY INSIGHT:
----------------
In proper RL, the agent must know:
1. WHERE it currently is (current architecture)
2. WHERE it wants to go (target specifications)

The state encodes BOTH the current configuration AND the target.
Actions modify the current configuration incrementally.
Rewards measure progress toward the target.

This is a GOAL-CONDITIONED RL problem.
"""

import numpy as np
from scipy.signal import find_peaks


class AntennaEnvOnline:
    """
    Proper RL Environment for Antenna Optimization.
    
    MDP Formulation:
    ----------------
    - State: [current_config (5), target_theta0 (1), current_metrics (3)] = 9D
             The agent knows its current configuration AND the goal.
    
    - Actions: Discrete modifications to the architecture
               0-4: Increment elements in ring 0-4
               5-9: Decrement elements in ring 0-4
               10: No-op (keep current configuration)
    
    - Reward: Improvement in RF metrics toward optimal performance
              + Bonus for achieving good SSL and HPBW
              - Penalty for too many elements (efficiency)
    
    - Episode: Agent tries to find optimal config for a random target angle
    """
    
    def __init__(self, max_rings=5, max_elements=10, max_steps=50, 
                 theta0_range=(30, 150), seed=None):
        """
        Args:
            max_rings: Number of concentric rings (default: 5)
            max_elements: Max elements per ring (default: 10)
            max_steps: Max steps per episode (default: 50)
            theta0_range: Range of pointing angles to train on
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        self.max_rings = max_rings
        self.max_elements = max_elements
        self.max_steps = max_steps
        self.theta0_range = theta0_range
        
        # Physical parameters for antenna simulation
        self.carrierFreq = 2.45e9  # 2.45 GHz
        self.c = 3e8  # Speed of light
        self.lambda_ = self.c / self.carrierFreq
        self.k = 2 * np.pi / self.lambda_
        self.r0 = 0.2 * self.lambda_  # First ring radius
        self.delta_r = 0.5 * self.lambda_  # Ring spacing
        
        # Action space: 11 discrete actions
        # 0-4: Add element to ring i
        # 5-9: Remove element from ring i  
        # 10: Keep current (no-op)
        self.n_actions = 2 * max_rings + 1
        
        # State normalization ranges
        self.ml_range = (0, 35)      # Main lobe gain in dB
        self.ssl_range = (-35, 0)    # Side lobe level in dB (negative is good)
        self.hpbw_range = (5, 180)   # HPBW in degrees
        
        # Episode state
        self.current_architecture = np.zeros(max_rings, dtype=np.int32)
        self.target_theta0 = 90.0
        self.t = 0
        self.best_reward_in_episode = -np.inf
        self.best_architecture_in_episode = None
        
    def compute_radiation_metrics(self, elements_per_ring, theta0_deg):
        """
        Compute RF metrics for a given antenna configuration using the pattern:
        - Gmax: peak of |AF| in dB
        - SSL: highest side-lobe level (dB), excluding a window around main lobe
        - HPBW: -3 dB beamwidth around the main lobe (degrees)
        """
        theta0 = np.deg2rad(theta0_deg)
        phi0 = 0
        phi = 0
        eps = 1e-10

        radii = self.r0 + self.delta_r * np.arange(self.max_rings)
        theta = np.linspace(0, 2 * np.pi, 1000)
        AF_az = np.zeros_like(theta, dtype=complex)

        total_elements = 0
        for ring in range(self.max_rings):
            a = radii[ring]
            N = int(elements_per_ring[ring])
            if N == 0:
                continue
            total_elements += N
            phi_n = 2 * np.pi * np.arange(N) / N
            for n in range(N):
                phase = self.k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                                      np.sin(theta0) * np.cos(phi0 - phi_n[n]))
                AF_az += np.exp(1j * phase)

        if total_elements == 0:
            return 0.0, -40.0, 180.0

        AF_abs = np.abs(AF_az)
        maxVal = np.max(AF_abs) + eps
        AF_norm = AF_abs / maxVal
        AF_dB = 20 * np.log10(AF_norm + eps)
        AF_dB = np.clip(AF_dB, -80, 0)
        theta_deg = np.rad2deg(theta)

        # Main lobe gain from pattern peak
        main_idx = int(np.argmax(AF_dB))
        main_lobe_gain = 20 * np.log10(maxVal + eps)

        # HPBW: -3 dB around main lobe
        threshold = AF_dB[main_idx] - 3.0
        # extend for circular wrap
        AF_ext = np.concatenate((AF_dB, AF_dB, AF_dB))
        theta_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
        main_ext_idx = main_idx + len(theta_deg)
        left_candidates = np.where(AF_ext[:main_ext_idx] <= threshold)[0]
        right_candidates = np.where(AF_ext[main_ext_idx:] <= threshold)[0]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            hpbw = 180.0
        else:
            left_idx = left_candidates[-1]
            right_idx = main_ext_idx + right_candidates[0]
            hpbw = float(abs(theta_ext[right_idx] - theta_ext[left_idx]))
            hpbw = min(hpbw, 180.0)

        # SSL: highest side-lobe outside main-lobe window
        main_window_deg = 10.0
        window = int(np.ceil(main_window_deg / 360.0 * len(theta_deg)))
        mask = np.ones_like(AF_dB, dtype=bool)
        mask[max(0, main_idx - window):min(len(theta_deg), main_idx + window + 1)] = False
        side_profile = np.where(mask, AF_dB, -80.0)
        peaks, _ = find_peaks(side_profile, distance=5)
        if len(peaks) == 0:
            ssl = -80.0
        else:
            ssl = float(np.max(side_profile[peaks]))

        return float(main_lobe_gain), float(ssl), float(hpbw)
    
    def _get_state(self):
        """
        Construct the state vector.
        
        State includes:
        - Current architecture (normalized): 5 values
        - Target pointing angle (normalized): 1 value  
        - Current RF metrics (normalized): 3 values
        
        Total: 9-dimensional state
        """
        # Normalize architecture (0 to max_elements -> 0 to 1)
        arch_norm = self.current_architecture / self.max_elements
        
        # Normalize target angle
        theta_norm = (self.target_theta0 - self.theta0_range[0]) / \
                     (self.theta0_range[1] - self.theta0_range[0])
        
        # Get current metrics
        ml, ssl, hpbw = self.compute_radiation_metrics(
            self.current_architecture, self.target_theta0
        )
        
        # Normalize metrics
        ml_norm = np.clip((ml - self.ml_range[0]) / (self.ml_range[1] - self.ml_range[0]), 0, 1)
        ssl_norm = np.clip((ssl - self.ssl_range[0]) / (self.ssl_range[1] - self.ssl_range[0]), 0, 1)
        hpbw_norm = np.clip((hpbw - self.hpbw_range[0]) / (self.hpbw_range[1] - self.hpbw_range[0]), 0, 1)
        
        state = np.array([
            *arch_norm,      # 5 values: current configuration
            theta_norm,      # 1 value: target angle
            ml_norm,         # 1 value: current gain
            ssl_norm,        # 1 value: current SSL
            hpbw_norm        # 1 value: current HPBW
        ], dtype=np.float32)
        
        return state
    
    def _compute_reward(self, ml, ssl, hpbw):
        """
        Reward aligned with evaluation:
        + w1 * Gmax_norm  (higher is better)
        - w2 * SSL_norm   (more negative SSL -> larger abs -> better)
        - w3 * HPBW_norm  (smaller HPBW -> better)
        - w4 * penalty_elements
        """
        eps = 1e-8
        total_elements = float(np.sum(self.current_architecture))
        if total_elements == 0:
            return -1.0

        gain_norm = np.clip(
            (ml - self.ml_range[0]) / (self.ml_range[1] - self.ml_range[0] + eps),
            0.0, 1.0
        )
        ssl_norm = np.clip(
            abs(ssl) / (abs(self.ssl_range[0]) + eps),
            0.0, 1.0
        )
        hpbw_norm = np.clip(
            (hpbw - self.hpbw_range[0]) / (self.hpbw_range[1] - self.hpbw_range[0] + eps),
            0.0, 1.0
        )
        penalty_elements = total_elements / (self.max_rings * self.max_elements)

        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
        reward = (
            w1 * gain_norm
            - w2 * ssl_norm
            - w3 * hpbw_norm
            - w4 * penalty_elements
        )
        return float(reward)
    
    def reset(self, target_theta0=None):
        """
        Reset environment for a new episode.
        
        Args:
            target_theta0: Specific pointing angle (random if None)
        
        Returns:
            Initial state vector
        """
        self.t = 0
        self.best_reward_in_episode = -np.inf
        self.best_architecture_in_episode = None
        
        # Set target angle
        if target_theta0 is not None:
            self.target_theta0 = float(target_theta0)
        else:
            self.target_theta0 = self.rng.uniform(*self.theta0_range)
        
        # Start with a random small configuration
        self.current_architecture = np.zeros(self.max_rings, dtype=np.int32)
        n_initial = self.rng.randint(1, 4)  # 1-3 initial elements
        for _ in range(n_initial):
            ring = self.rng.randint(0, self.max_rings)
            self.current_architecture[ring] = min(
                self.current_architecture[ring] + 1,
                self.max_elements
            )
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute action and return new state.
        
        Actions:
            0-4: Add element to ring i
            5-9: Remove element from ring i
            10: No operation
        
        Returns:
            next_state, reward, done, info
        """
        self.t += 1
        action = int(action)
        
        # Apply action
        if action < self.max_rings:
            # Add element to ring
            ring_idx = action
            self.current_architecture[ring_idx] = min(
                self.current_architecture[ring_idx] + 1,
                self.max_elements
            )
        elif action < 2 * self.max_rings:
            # Remove element from ring
            ring_idx = action - self.max_rings
            self.current_architecture[ring_idx] = max(
                self.current_architecture[ring_idx] - 1,
                0
            )
        # else: no-op (action == 10)
        
        # Ensure at least one element
        if np.sum(self.current_architecture) == 0:
            self.current_architecture[self.rng.randint(0, self.max_rings)] = 1
        
        # Compute metrics and reward
        ml, ssl, hpbw = self.compute_radiation_metrics(
            self.current_architecture, self.target_theta0
        )
        reward = self._compute_reward(ml, ssl, hpbw)
        
        # Track best in episode
        if reward > self.best_reward_in_episode:
            self.best_reward_in_episode = reward
            self.best_architecture_in_episode = self.current_architecture.copy()
        
        # Episode termination
        done = self.t >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'main_lobe_gain': ml,
            'ssl': ssl,
            'hpbw': hpbw,
            'architecture': self.current_architecture.copy(),
            'theta0': self.target_theta0,
            'total_elements': int(np.sum(self.current_architecture)),
            'best_architecture': self.best_architecture_in_episode,
            'best_reward': self.best_reward_in_episode
        }
        
        return next_state, reward, done, info
    
    @property
    def state_dim(self):
        """State dimension: 5 (arch) + 1 (theta0) + 3 (metrics) = 9"""
        return 9
    
    @property
    def action_dim(self):
        """Action dimension: 5 (add) + 5 (remove) + 1 (no-op) = 11"""
        return self.n_actions
    
    def get_best_configuration(self, theta0_deg, n_random=20000):
        """
        Find the best configuration through random search (for comparison).
        
        This is NOT how the agent learns - it's just for evaluation baseline.
        """
        best_reward = -np.inf
        best_config = None
        best_metrics = None
        
        for _ in range(n_random):
            # Random configuration
            config = self.rng.randint(0, self.max_elements + 1, size=self.max_rings)
            if np.sum(config) == 0:
                config[self.rng.randint(0, self.max_rings)] = 1
            
            ml, ssl, hpbw = self.compute_radiation_metrics(config, theta0_deg)
            
            # Same reward function
            self.current_architecture = config
            reward = self._compute_reward(ml, ssl, hpbw)
            
            if reward > best_reward:
                best_reward = reward
                best_config = config.copy()
                best_metrics = (ml, ssl, hpbw)
        
        return best_config, best_metrics, best_reward
