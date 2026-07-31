import os
import numpy as np
import matplotlib.pyplot as plt
from environment import MultiVehicleIntersectionEnv
from agents import WolfAgent, QLearningAgent
from utils import run_experiment

QUEUE_DEPTHS = [1, 2, 3, 4]
EPISODES = 2500
N_SEEDS = 3

def main():
    """
    Main execution pipeline for evaluating multi-agent reinforcement learning 
    scalability (4, 8, 12, and 16 agents) across symmetric and emergency preemption scenarios.
    """
    # Ensure relative path output directory exists for figures
    os.makedirs("../figures", exist_ok=True)

    # Experimental configuration parameters
    EPISODES = 1000
    SEEDS = [42, 101, 202]  # Multiple random seeds for statistical stability (error bars)
    QUEUE_DEPTHS = [1, 2, 3, 4]  # Corresponding to 4, 8, 12, and 16 total agents (4 lanes x Q)
    
    # Define experiment scenarios for scalability benchmark
    scenarios = [
        ("pure", "Pure Scalability (no priority)"),
        ("ambulance_random", "Scalability + Random Ambulance")
    ]

    for mode, title in scenarios:
        print(f"\n==========================================")
        print(f" Running Scalability Benchmark: {title}")
        print(f"==========================================")

        # Storage structures for aggregated metrics across queue depths
        wolf_success_means, wolf_success_stds = [], []
        q_success_means, q_success_stds = [], []
        
        wolf_crash_means, wolf_crash_stds = [], []
        q_crash_means, q_crash_stds = [], []

        for q in QUEUE_DEPTHS:
            total_agents = 4 * q
            print(f"Testing queue depth Q={q} ({total_agents} simultaneous agents)...")

            # Temporary arrays to hold seed results per agent count
            w_succ_seeds, q_succ_seeds = [], []
            w_crash_seeds, q_crash_seeds = [], []

            for s in SEEDS:
                # 1. Run WoLF-PHC policy evaluation
                w_crash, w_succ = run_experiment("wolf", queue_depth=q, episodes=EPISODES, mode=mode, seed=s)
                # Compute average performance over the last 100 episodes (converged policy)
                w_succ_seeds.append(np.mean(w_succ[-100:]))
                w_crash_seeds.append(np.mean(w_crash[-100:]))

                # 2. Run Independent Q-Learning baseline evaluation
                q_crash, q_succ = run_experiment("q_learning", queue_depth=q, episodes=EPISODES, mode=mode, seed=s)
                q_succ_seeds.append(np.mean(q_succ[-100:]))
                q_crash_seeds.append(np.mean(q_crash[-100:]))

            # Calculate statistical mean and standard deviation across random seeds
            wolf_success_means.append(np.mean(w_succ_seeds))
            wolf_success_stds.append(np.std(w_succ_seeds))
            q_success_means.append(np.mean(q_succ_seeds))
            q_success_stds.append(np.std(q_succ_seeds))

            wolf_crash_means.append(np.mean(w_crash_seeds))
            wolf_crash_stds.append(np.std(w_crash_seeds))
            q_crash_means.append(np.mean(q_crash_seeds))
            q_crash_stds.append(np.std(q_crash_seeds))

        # -----------------------------------------------------------------
        # Visualization: Generate Scalability Comparison Plots
        # -----------------------------------------------------------------
        agent_counts = [4 * q for q in QUEUE_DEPTHS]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Success Rate vs. Agent Count
        axes[0].errorbar(agent_counts, wolf_success_means, yerr=wolf_success_stds, 
                         fmt='-o', color='#2ca02c', label='WoLF-PHC', capsize=4, linewidth=2)
        axes[0].errorbar(agent_counts, q_success_means, yerr=q_success_stds, 
                         fmt='-s', color='#d62728', label='Independent Q-Learning', capsize=4, linewidth=2)
        axes[0].set_title(f"{title}\nSuccess Rate vs. Agent Count", fontweight="bold")
        axes[0].set_xlabel("Total simultaneous agents (4 lanes x Q)")
        axes[0].set_ylabel("Success rate")
        axes[0].set_xticks(agent_counts)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        # Plot 2: Crash Rate vs. Agent Count
        axes[1].errorbar(agent_counts, wolf_crash_means, yerr=wolf_crash_stds, 
                         fmt='-o', color='#2ca02c', label='WoLF-PHC', capsize=4, linewidth=2)
        axes[1].errorbar(agent_counts, q_crash_means, yerr=q_crash_stds, 
                         fmt='-s', color='#d62728', label='Independent Q-Learning', capsize=4, linewidth=2)
        axes[1].set_title(f"{title}\nCrash Rate vs. Agent Count", fontweight="bold")
        axes[1].set_xlabel("Total simultaneous agents (4 lanes x Q)")
        axes[1].set_ylabel("Crash rate")
        axes[1].set_xticks(agent_counts)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        plt.tight_layout()

        # Save plot figure
        filename = "multi_scenario_comparison.png" if mode == "pure" else "scalability_ambulance_comparison.png"
        fig_path = os.path.join("../figures", filename)
        plt.savefig(fig_path, dpi=150)
        print(f" Scalability figure saved successfully to: {fig_path}")
        plt.close()

if __name__ == "__main__":
    main()