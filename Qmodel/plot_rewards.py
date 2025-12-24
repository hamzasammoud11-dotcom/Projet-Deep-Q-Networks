"""
plot_rewards.py - Visualisation de l'historique d'entraînement
================================================================
Ce script charge et affiche la courbe d'apprentissage du DQN.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")


def main():
    clear_console()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Charger l'historique
    reward_file = os.path.join(script_dir, 'reward_history.npy')
    
    if not os.path.exists(reward_file):
        print("Error: reward_history.npy file not found.")
        print("   Please run train_dqn.py first.")
        return
    
    reward_history = np.load(reward_file)
    print(f"Historique chargé: {len(reward_history)} épisodes")
    
    # Statistiques
    print(f"\n=== Statistiques ===")
    print(f"Reward moyen: {np.mean(reward_history):.4f}")
    print(f"Reward max: {np.max(reward_history):.4f}")
    print(f"Reward min: {np.min(reward_history):.4f}")
    print(f"Écart-type: {np.std(reward_history):.4f}")
    
    # Premiers et derniers épisodes
    n = min(100, len(reward_history) // 4)
    if n > 0:
        print(f"\nMoyenne des {n} premiers épisodes: {np.mean(reward_history[:n]):.4f}")
        print(f"Moyenne des {n} derniers épisodes: {np.mean(reward_history[-n:]):.4f}")
    
    # Tracé
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Graphique 1: Reward par épisode avec moyenne mobile
    ax1 = axes[0]
    ax1.plot(reward_history, alpha=0.3, color='blue', label='Reward par épisode')
    
    # Moyenne mobile
    windows = [50, 100]
    colors = ['orange', 'red']
    for window, color in zip(windows, colors):
        if len(reward_history) >= window:
            moving_avg = np.convolve(reward_history, 
                                      np.ones(window)/window, 
                                      mode='valid')
            ax1.plot(range(window-1, len(reward_history)), moving_avg, 
                     label=f'Moyenne mobile ({window})', linewidth=2, color=color)
    
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Récompense cumulée")
    ax1.set_title("Courbe d'apprentissage DQN")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Distribution des rewards
    ax2 = axes[1]
    ax2.hist(reward_history, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(reward_history), color='red', linestyle='--', 
                linewidth=2, label=f'Moyenne: {np.mean(reward_history):.3f}')
    ax2.set_xlabel("Récompense")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des récompenses")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'reward_analysis.png'), dpi=300)
    plt.show()
    
    print("\n✅ Figure sauvegardée: reward_analysis.png")


if __name__ == "__main__":
    main()
