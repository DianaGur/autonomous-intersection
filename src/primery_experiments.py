import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import MultiVehicleIntersectionEnv
from agents import WolfAgent, QLearningAgent

def run_experiment_with_dataset(agent_type, queue_depth, episodes, mode, seed=42):
    """
    Executes a single reinforcement learning experiment, logs trajectory 
    data for dataset generation, and tracks metrics for convergence analysis.

    Parameters:
        agent_type (str): Type of agent ('wolf' or 'q_learning').
        queue_depth (int): Queue length capacity per approach lane.
        episodes (int): Total training episodes to run.
        mode (str): Environment scenario ('pure' or 'ambulance_random').
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (crash_hist, success_hist, reward_hist, dataset_records)
    """
    
    np.random.seed(seed)
    
    # Initialize the intersection environment and active RL agents
    env = MultiVehicleIntersectionEnv(queue_depth, mode=mode)
    n = env.n_agents
    agents = {i: (WolfAgent() if agent_type == "wolf" else QLearningAgent()) for i in range(n)}
    max_steps = 4 * (queue_depth + 2)

    # Tracking lists for historical analytics
    crash_hist, success_hist, reward_hist = [], [], []
    dataset_records = []

    for ep in range(episodes):
        state = env.reset()
        steps = 0
        ep_reward = 0  # Total collective reward accumulated during this episode

        while not all(env.done.values()) and not env.crash and steps < max_steps:
            # Decay epsilon linearly over 70% of training episodes down to a floor of 0.02
            epsilon = max(0.02, 0.3 * (1 - ep / (episodes * 0.7)))
            actions = {i: agents[i].get_action(state, epsilon) for i in range(n)}
            
            # Prepare row entry for tabular dataset recording
            record = {
                "episode": ep,
                "step": steps,
                "agent_type": agent_type,
                "mode": mode,
                "state": str(state),
                "crash": env.crash
            }
            for i in range(n):
                record[f"agent_{i}_pos"] = env.pos[i]
                record[f"agent_{i}_action"] = actions[i]

            # Transition step in the environment
            next_state, rewards, done = env.step(actions)
            
            # Sum up collective system reward for this time step
            step_system_reward = sum(rewards.values())
            ep_reward += step_system_reward

            # Agent policy update & logging reward per agent
            for i in range(n):
                agents[i].learn(state, actions[i], rewards[i], next_state)
                record[f"agent_{i}_reward"] = rewards[i]

            dataset_records.append(record)
            state = next_state
            steps += 1

        # Episode-level performance metrics logging
        crash_hist.append(1 if env.crash else 0)
        success_hist.append(1 if all(env.done.values()) and not env.crash else 0)
        reward_hist.append(ep_reward)

    return crash_hist, success_hist, reward_hist, dataset_records

def moving_average(data, window_size=30):
    """Computes moving average over a given window to smooth noisy RL curves."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    # Ensure relative output directories exist
    os.makedirs("../figures", exist_ok=True)
    os.makedirs("../dataset", exist_ok=True)

    EPISODES = 1500
    window = 30
    all_dataset_rows = []

    # Experimental scenario configuration mapping
    scenarios = [
        ("pure", "Scenario 1: 2 Standard Vehicles (Pure)", "scenario_1_reward_analysis.png"),
        ("ambulance_random", "Scenario 2: 2 Vehicles + Emergency Vehicle", "scenario_2_reward_analysis.png")
    ]

    # Initialize subplots figure for the split Success Rate convergence visualization
    fig_succ, axes_succ = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (mode, title, reward_filename) in enumerate(scenarios):
        print(f"\n--- Running Experiment for: {title} ---")
        
        # 1. Run WoLF-PHC algorithm
        _, w_succ, w_reward, w_data = run_experiment_with_dataset("wolf", queue_depth=1, episodes=EPISODES, mode=mode)
        all_dataset_rows.extend(w_data)
        
        # 2. Run Independent Q-Learning baseline algorithm
        _, q_succ, q_reward, q_data = run_experiment_with_dataset("q_learning", queue_depth=1, episodes=EPISODES, mode=mode)
        all_dataset_rows.extend(q_data)

        # Smooth raw metrics using moving average
        w_succ_smooth = moving_average(w_succ, window)
        q_succ_smooth = moving_average(q_succ, window)
        
        w_reward_smooth = moving_average(w_reward, window)
        q_reward_smooth = moving_average(q_reward, window)

        # -----------------------------------------------------------------
        # Plotting & Formatting: Success Rate Subplots
        # -----------------------------------------------------------------
        axes_succ[idx].plot(w_succ_smooth, label="WoLF-PHC", color="#2ca02c", linewidth=2)
        axes_succ[idx].plot(q_succ_smooth, label="Independent Q-Learning", color="#d62728", linewidth=2)
        axes_succ[idx].set_title(title, fontweight="bold")
        axes_succ[idx].set_xlabel("Episode")
        axes_succ[idx].set_ylabel("Success Rate (Moving Avg)")
        axes_succ[idx].grid(True, linestyle="--", alpha=0.5)
        axes_succ[idx].legend(loc="upper left")

        # -----------------------------------------------------------------
        # Plotting & Saving: Individual System Reward Analysis Figures
        # -----------------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(w_reward_smooth, label="WoLF-PHC (Variable Learning Rates)", color="#2ca02c", linewidth=2)
        plt.plot(q_reward_smooth, label="Independent Q-Learning (Baseline)", color="#d62728", linewidth=2, linestyle="--")
        plt.title(f"Comparative Analysis: {title}", fontsize=13, fontweight="bold")
        plt.xlabel("Training Episodes", fontsize=11)
        plt.ylabel("Smoothed Collective System Reward", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        reward_fig_path = os.path.join("../figures", reward_filename)
        plt.savefig(reward_fig_path, dpi=150)
        print(f" Saved: {reward_fig_path}")
        plt.close()

    # -----------------------------------------------------------------
    # Exporting Split Success Rate Figure
    # -----------------------------------------------------------------
    fig_succ.tight_layout()
    succ_fig_path = os.path.join("../figures", "small_experiments_convergence.png")
    fig_succ.savefig(succ_fig_path, dpi=150)
    print(f" Saved: {succ_fig_path}")
    plt.close(fig_succ)

    # -----------------------------------------------------------------
    # Exporting Simulation Trajectory Dataset (CSV)
    # -----------------------------------------------------------------
    df = pd.DataFrame(all_dataset_rows)
    csv_path = os.path.join("../dataset", "simulation_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Dataset successfully exported to: {csv_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()