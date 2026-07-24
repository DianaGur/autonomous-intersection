import numpy as np
import matplotlib.pyplot as plt
from utils import run_experiment, last_window_mean

QUEUE_DEPTHS = [1, 2, 3, 4]
EPISODES = 2500
N_SEEDS = 3

def main():
    all_results = {}
    for mode in ["pure", "ambulance_random"]:
        print(f"\n=========== MODE: {mode} ===========")
        results = {"wolf": {}, "q_learning": {}}
        for agent_type in ["wolf", "q_learning"]:
            for q in QUEUE_DEPTHS:
                succ_finals, crash_finals = [], []
                for seed in range(N_SEEDS):
                    crash_h, succ_h = run_experiment(agent_type, q, EPISODES, seed, mode)
                    succ_finals.append(last_window_mean(succ_h))
                    crash_finals.append(last_window_mean(crash_h))
                results[agent_type][q] = {
                    "success_mean": np.mean(succ_finals), "success_std": np.std(succ_finals),
                    "crash_mean": np.mean(crash_finals), "crash_std": np.std(crash_finals),
                }
                print(f"{agent_type:12s} | Q={q} ({4*q:2d} agents) | "
                      f"success={results[agent_type][q]['success_mean']:.3f}±{results[agent_type][q]['success_std']:.3f} | "
                      f"crash={results[agent_type][q]['crash_mean']:.3f}±{results[agent_type][q]['crash_std']:.3f}")
        all_results[mode] = results

    n_agents_axis = [4 * q for q in QUEUE_DEPTHS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for row, mode in enumerate(["pure", "ambulance_random"]):
        results = all_results[mode]
        for agent_type, color, marker in [("wolf", "#2ca02c", "o"), ("q_learning", "#d62728", "s")]:
            label = "WoLF-PHC" if agent_type == "wolf" else "Independent Q-Learning"
            succ = [results[agent_type][q]["success_mean"] for q in QUEUE_DEPTHS]
            succ_err = [results[agent_type][q]["success_std"] for q in QUEUE_DEPTHS]
            axes[row, 0].errorbar(n_agents_axis, succ, yerr=succ_err, color=color, marker=marker,
                                   linewidth=2, capsize=4, label=label)

            crash = [results[agent_type][q]["crash_mean"] for q in QUEUE_DEPTHS]
            crash_err = [results[agent_type][q]["crash_std"] for q in QUEUE_DEPTHS]
            axes[row, 1].errorbar(n_agents_axis, crash, yerr=crash_err, color=color, marker=marker,
                                   linewidth=2, capsize=4, label=label)

        title_prefix = "Pure Scalability (no priority)" if mode == "pure" else "Scalability + Random Ambulance"
        axes[row, 0].set_title(f"{title_prefix}\nSuccess Rate vs. Agent Count", fontweight="bold")
        axes[row, 1].set_title(f"{title_prefix}\nCrash Rate vs. Agent Count", fontweight="bold")
        for c in [0, 1]:
            axes[row, c].set_xlabel("Total simultaneous agents (4 lanes x Q)")
            axes[row, c].set_xticks(n_agents_axis)
            axes[row, c].set_ylim(-0.02, 1.05)
            axes[row, c].grid(True, linestyle="--", alpha=0.5)
            axes[row, c].legend()
        axes[row, 0].set_ylabel("Success rate")
        axes[row, 1].set_ylabel("Crash rate")

    plt.tight_layout()
    plt.savefig("../figures/multi_scenario_comparison.png", dpi=150)
    print("\nGraph saved successfully to figures/multi_scenario_comparison.png!")
    plt.show()

if __name__ == "__main__":
    main()