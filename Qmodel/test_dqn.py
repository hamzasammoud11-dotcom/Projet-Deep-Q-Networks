"""
test_dqn.py - Evaluation of Trained DQN Agent
==============================================

This script evaluates the trained DQN agent **without any dataset** by:
1. Testing on multiple pointing angles
2. Comparing the DQN's chosen architecture to the best architecture found
   by an independent search (random search baseline)
3. Computing RF metrics (Gmax, SSL, HPBW) and a weighted prediction error
   (DQN vs best-found architecture)
4. Generating radiation pattern visualizations

Prediction error (what the professor asked):
- Reference = best architecture found by search for the same angle (no dataset)
- Error per metric = |metric_dqn - metric_ref|
- Relative error = normalized by expected range (Gmax: 35 dB, SSL: 35 dB, HPBW: 180 deg)
- Global error = 0.4*rel_Gmax + 0.3*rel_SSL + 0.3*rel_HPBW (percent)

Output files:
- test_polar.png: Polar radiation pattern plot
- test_results.npy/.mat: Detailed results matrix
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from scipy.io import savemat

from env_antenna import AntennaEnvOnline


def clear_console():
    print("\n" + "=" * 80)
    print(" " * 25 + "DQN ANTENNA EVALUATION")
    print("=" * 80 + "\n")


# =============================================================================
# Q-Network (must match train_dqn.py)
# =============================================================================
class QNetwork(nn.Module):
    """Same architecture as in training."""
    
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# Radiation Pattern Computation
# =============================================================================
def compute_radiation_pattern(elements_per_ring, theta0_deg, n_points=1000):
    """
    Compute full radiation pattern for visualization.
    
    Args:
        elements_per_ring: Array of element counts per ring
        theta0_deg: Pointing angle in degrees
        n_points: Number of angle samples
    
    Returns:
        theta_rad: Angles in radians
        theta_deg: Angles in degrees
        AF_dB: Normalized pattern in dB
        metrics: (main_lobe_gain, ssl, hpbw)
    """
    # Physical parameters
    c = 3e8
    f = 2.45e9
    lambda_ = c / f
    k = 2 * np.pi / lambda_
    r0 = 0.2 * lambda_
    delta_r = 0.5 * lambda_
    max_rings = 5
    
    theta0 = np.deg2rad(theta0_deg)
    phi0 = 0
    phi = 0
    
    radii = r0 + delta_r * np.arange(max_rings)
    theta = np.linspace(0, 2 * np.pi, n_points)
    AF = np.zeros_like(theta, dtype=complex)
    
    total_elements = 0
    for ring in range(min(max_rings, len(elements_per_ring))):
        a = radii[ring]
        N = int(elements_per_ring[ring])
        if N == 0:
            continue
        total_elements += N
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF += np.exp(1j * phase)
    
    if total_elements == 0:
        return theta, np.rad2deg(theta), np.full_like(theta, -80), (0.0, -80.0, 180.0)
    
    AF_abs = np.abs(AF)
    maxVal = np.max(AF_abs) + 1e-10
    AF_norm = AF_abs / maxVal
    AF_dB = 20 * np.log10(AF_norm + 1e-10)
    AF_dB = np.clip(AF_dB, -80, 0)
    theta_deg = np.rad2deg(theta)
    
    main_idx = int(np.argmax(AF_dB))
    main_lobe_gain = 20 * np.log10(maxVal + 1e-10)
    
    threshold = AF_dB[main_idx] - 3.0
    AF_dB_ext = np.concatenate((AF_dB, AF_dB, AF_dB))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    main_ext_idx = main_idx + len(theta_deg)
    left_candidates = np.where(AF_dB_ext[:main_ext_idx] <= threshold)[0]
    right_candidates = np.where(AF_dB_ext[main_ext_idx:] <= threshold)[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        hpbw = 180.0
    else:
        left_idx = left_candidates[-1]
        right_idx = main_ext_idx + right_candidates[0]
        hpbw = float(abs(theta_deg_ext[right_idx] - theta_deg_ext[left_idx]))
        hpbw = min(hpbw, 180.0)
    
    main_window_deg = 10.0
    window = int(np.ceil(main_window_deg / 360.0 * len(theta_deg)))
    mask = np.ones_like(AF_dB, dtype=bool)
    mask[max(0, main_idx - window):min(len(theta_deg), main_idx + window + 1)] = False
    side_profile = np.where(mask, AF_dB, -80.0)
    peaks, _ = find_peaks(side_profile, distance=5)
    ssl = float(np.max(side_profile[peaks])) if len(peaks) > 0 else -80.0
    
    return theta, theta_deg, AF_dB, (main_lobe_gain, ssl, hpbw)


def compute_prediction_error(pred_metrics, ref_metrics, ranges):
    """
    Compute prediction error vs a reference architecture (no dataset).
    
    Args:
        pred_metrics: (ml, ssl, hpbw) from DQN-chosen config
        ref_metrics: (ml, ssl, hpbw) from best-found config (search baseline)
        ranges: dict with normalizing ranges: {'ml': float, 'ssl': float, 'hpbw': float}
    Returns:
        dict with absolute and relative errors + weighted global error
    """
    ml_pred, ssl_pred, hpbw_pred = pred_metrics
    ml_ref, ssl_ref, hpbw_ref = ref_metrics
    
    mae_ml = abs(ml_pred - ml_ref)
    mae_ssl = abs(ssl_pred - ssl_ref)
    mae_hpbw = abs(hpbw_pred - hpbw_ref)
    
    rel_ml = mae_ml / (abs(ranges['ml']) + 1e-8)
    rel_ssl = mae_ssl / (abs(ranges['ssl']) + 1e-8)
    rel_hpbw = mae_hpbw / (abs(ranges['hpbw']) + 1e-8)
    
    w1, w2, w3 = 0.4, 0.3, 0.3  # Emphasize Gmax slightly
    error_global = (w1 * rel_ml + w2 * rel_ssl + w3 * rel_hpbw) * 100
    
    return {
        'mae_ml': mae_ml,
        'mae_ssl': mae_ssl,
        'mae_hpbw': mae_hpbw,
        'rel_ml': rel_ml * 100,
        'rel_ssl': rel_ssl * 100,
        'rel_hpbw': rel_hpbw * 100,
        'error_global': error_global
    }


# =============================================================================
# DQN Evaluation
# =============================================================================
def evaluate_dqn(model, device, env, theta0_deg, max_steps=50):
    """
    Evaluate DQN agent for a specific pointing angle.
    
    The agent runs an episode using its learned policy (no exploration).
    Returns the best configuration found during the episode.
    
    Args:
        model: Trained Q-network
        device: Torch device
        env: Antenna environment
        theta0_deg: Target pointing angle
        max_steps: Maximum steps to run
    
    Returns:
        best_config: Best architecture found
        best_metrics: (ml, ssl, hpbw) of best config
        trajectory: List of (config, metrics, reward) tuples
    """
    state = env.reset(target_theta0=theta0_deg)
    
    trajectory = []
    best_reward = -np.inf
    best_config = None
    best_metrics = None
    
    for step in range(max_steps):
        # Greedy action selection (no exploration)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Track trajectory
        trajectory.append({
            'step': step,
            'architecture': info['architecture'].copy(),
            'metrics': (info['main_lobe_gain'], info['ssl'], info['hpbw']),
            'reward': reward,
            'total_elements': info['total_elements']
        })
        
        # Track best
        if reward > best_reward:
            best_reward = reward
            best_config = info['architecture'].copy()
            best_metrics = (info['main_lobe_gain'], info['ssl'], info['hpbw'])
        
        state = next_state
        
        if done:
            break
    
    return best_config, best_metrics, trajectory


def load_model(model_path, state_dim=9, action_dim=11):
    """Load trained DQN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QNetwork(state_dim, action_dim, hidden1=256, hidden2=128).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['q_network'])
    model.eval()
    
    return model, device


# =============================================================================
# Main Evaluation
# =============================================================================
def main():
    clear_console()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'dqn_model.pth')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("ERROR: Model not found!")
        print("Please run train_dqn.py first to train the model.")
        return
    
    # Create environment
    env = AntennaEnvOnline(max_rings=5, max_elements=10, max_steps=50, seed=123)
    
    # Load model
    model, device = load_model(model_path, state_dim=env.state_dim, action_dim=env.action_dim)
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # Test angles
    test_angles = [30, 60, 90, 120, 150]
    
    print("\n" + "=" * 70)
    print("EVALUATING DQN AGENT")
    print("=" * 70)
    
    results = []
    ranges = {
        'ml': env.ml_range[1],
        'ssl': abs(env.ssl_range[0]),
        'hpbw': env.hpbw_range[1]
    }
    
    for theta0 in test_angles:
        print(f"\n--- Pointing angle: theta0 = {theta0} deg ---")
        
        # DQN prediction
        dqn_config, dqn_metrics, trajectory = evaluate_dqn(
            model, device, env, theta0, max_steps=50
        )
        ml_dqn, ssl_dqn, hpbw_dqn = dqn_metrics
        
        print(f"\nDQN Result:")
        print(f"  Architecture: {dqn_config}")
        print(f"  Total elements: {np.sum(dqn_config)}")
        print(f"  Gmax: {ml_dqn:.2f} dB")
        print(f"  SSL:  {ssl_dqn:.2f} dB")
        print(f"  HPBW: {hpbw_dqn:.2f} deg")
        
        # Random search baseline (massive search, no dataset)
        baseline_config, baseline_metrics, baseline_reward = env.get_best_configuration(
            theta0, n_random=20000
        )
        ml_base, ssl_base, hpbw_base = baseline_metrics
        
        print(f"\nRandom Search Baseline (500 samples):")
        print(f"  Architecture: {baseline_config}")
        print(f"  Total elements: {np.sum(baseline_config)}")
        print(f"  Gmax: {ml_base:.2f} dB")
        print(f"  SSL:  {ssl_base:.2f} dB")
        print(f"  HPBW: {hpbw_base:.2f} deg")
        
        # Compute prediction errors (DQN vs best-found baseline)
        errors = compute_prediction_error(
            (ml_dqn, ssl_dqn, hpbw_dqn),
            (ml_base, ssl_base, hpbw_base),
            ranges
        )
        
        print(f"\nPrediction errors (DQN vs best-found):")
        print(f"  |Gmax| error:  {errors['mae_ml']:.2f} dB ({errors['rel_ml']:.2f}%)")
        print(f"  |SSL|  error:  {errors['mae_ssl']:.2f} dB ({errors['rel_ssl']:.2f}%)")
        print(f"  |HPBW| error:  {errors['mae_hpbw']:.2f} deg ({errors['rel_hpbw']:.2f}%)")
        print(f"  Global weighted error: {errors['error_global']:.2f}%")
        
        # Store results
        results.append({
            'theta0': theta0,
            'dqn_config': dqn_config,
            'dqn_metrics': dqn_metrics,
            'baseline_config': baseline_config,
            'baseline_metrics': baseline_metrics,
            'errors': errors
        })
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    avg_error = np.mean([r['errors']['error_global'] for r in results])
    std_error = np.std([r['errors']['error_global'] for r in results])
    
    print(f"\nAverage global error (DQN vs best-found): {avg_error:.2f}% +/- {std_error:.2f}%")
    
    # ==========================================================================
    # Visualization: Polar radiation pattern
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    theta0_viz = 90  # Use 90 degrees for visualization
    
    # Get DQN result for visualization angle
    dqn_config_viz, dqn_metrics_viz, _ = evaluate_dqn(
        model, device, env, theta0_viz, max_steps=50
    )
    baseline_config_viz, baseline_metrics_viz, _ = env.get_best_configuration(
        theta0_viz, n_random=20000
    )
    
    # Compute radiation patterns
    theta_rad, theta_deg, AF_dB_dqn, _ = compute_radiation_pattern(dqn_config_viz, theta0_viz)
    _, _, AF_dB_baseline, _ = compute_radiation_pattern(baseline_config_viz, theta0_viz)
    
    # Polar plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    ax.plot(theta_rad, AF_dB_dqn, 'b-', linewidth=2, 
            label=f'DQN: {dqn_config_viz}')
    ax.plot(theta_rad, AF_dB_baseline, 'r--', linewidth=2, 
            label=f'Baseline: {baseline_config_viz}')
    
    # Mark pointing direction
    ax.axvline(np.deg2rad(theta0_viz), color='green', linestyle=':', 
               linewidth=2, label=f'Target: {theta0_viz} deg')
    
    ax.set_title(f'Radiation Pattern Comparison (theta0 = {theta0_viz} deg)', 
                 fontsize=14, pad=20)
    ax.set_rlim([-40, 0])
    ax.set_rticks([-40, -30, -20, -10, 0])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'test_polar.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFigure saved: test_polar.png")
    
    # ==========================================================================
    # Save results to files
    # ==========================================================================
    
    # Prepare results matrix
    columns = [
        'theta0_deg',
        'ring1', 'ring2', 'ring3', 'ring4', 'ring5',
        'gmax_dqn', 'ssl_dqn', 'hpbw_dqn',
        'gmax_baseline', 'ssl_baseline', 'hpbw_baseline',
        'mae_gmax', 'mae_ssl', 'mae_hpbw',
        'rel_gmax_percent', 'rel_ssl_percent', 'rel_hpbw_percent',
        'error_global_percent'
    ]
    
    results_matrix = []
    for r in results:
        row = [
            float(r['theta0']),
            *[int(x) for x in r['dqn_config']],
            *[float(x) for x in r['dqn_metrics']],
            *[float(x) for x in r['baseline_metrics']],
            float(r['errors']['mae_ml']),
            float(r['errors']['mae_ssl']),
            float(r['errors']['mae_hpbw']),
            float(r['errors']['rel_ml']),
            float(r['errors']['rel_ssl']),
            float(r['errors']['rel_hpbw']),
            float(r['errors']['error_global'])
        ]
        results_matrix.append(row)
    
    results_matrix = np.array(results_matrix, dtype=np.float32)
    
    # Save as NPY
    np.save(os.path.join(script_dir, 'test_results.npy'), results_matrix)
    
    # Save as MAT
    savemat(os.path.join(script_dir, 'test_results.mat'), {
        'results_matrix': results_matrix,
        'columns': np.array(columns, dtype=object),
        'summary': {
            'avg_error_global': avg_error,
            'std_error_global': std_error
        }
    })
    
    print("Results saved: test_results.npy, test_results.mat")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
