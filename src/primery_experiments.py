import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import MultiVehicleIntersectionEnv
from agents import WolfAgent, QLearningAgent

def run_experiment_with_dataset(agent_type, queue_depth, episodes, mode, seed=42):
    np.random.seed(seed)
    env = MultiVehicleIntersectionEnv(queue_depth, mode=mode)
    n = env.n_agents
    agents = {i: (WolfAgent() if agent_type == "wolf" else QLearningAgent()) for i in range(n)}
    max_steps = 4 * (queue_depth + 2)

    crash_hist, success_hist, reward_hist = [], [], []
    dataset_records = []

    for ep in range(episodes):
        state = env.reset()
        steps = 0
        ep_reward = 0  # Initialize episode reward

        while not all(env.done.values()) and not env.crash and steps < max_steps:
            epsilon = max(0.02, 0.3 * (1 - ep / (episodes * 0.7)))
            actions = {i: agents[i].get_action(state, epsilon) for i in range(n)}
            
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

            next_state, rewards, done = env.step(actions)
            
            step_system_reward = sum(rewards.values())
            ep_reward += step_system_reward

            for i in range(n):
                agents[i].learn(state, actions[i], rewards[i], next_state)
                record[f"agent_{i}_reward"] = rewards[i]

            dataset_records.append(record)
            state = next_state
            steps += 1

        crash_hist.append(1 if env.crash else 0)
        success_hist.append(1 if all(env.done.values()) and not env.crash else 0)
        reward_hist.append(ep_reward)

    return crash_hist, success_hist, reward_hist, dataset_records

def moving_average(data, window_size=30):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():

    EPISODES = 1500
    all_dataset_rows = []

    scenarios = [
        ("pure", "Scenario 1: 2 Standard Vehicles (Pure)", "scenario_1_reward_analysis.png"),
        ("ambulance_random", "Scenario 2: 2 Vehicles + Emergency Vehicle", "scenario_2_reward_analysis.png")
    ]

    for mode, title, filename in scenarios:
        print(f"\n--- Running Reward Experiment for: {title} ---")
        
        # WoLF-PHC
        _, _, w_reward, w_data = run_experiment_with_dataset("wolf", queue_depth=1, episodes=EPISODES, mode=mode)
        all_dataset_rows.extend(w_data)
        
        # Q-Learning
        _, _, q_reward, q_data = run_experiment_with_dataset("q_learning", queue_depth=1, episodes=EPISODES, mode=mode)
        all_dataset_rows.extend(q_data)

        # Smoothing the rewards for better visualization
        window = 30
        w_reward_smooth = moving_average(w_reward, window)
        q_reward_smooth = moving_average(q_reward, window)

        # Plotting the smoothed rewards
        plt.figure(figsize=(10, 5))
        plt.plot(w_reward_smooth, label="WoLF-PHC (Variable Learning Rates)", color="#2ca02c", linewidth=2)
        plt.plot(q_reward_smooth, label="Independent Q-Learning (Baseline)", color="#d62728", linewidth=2, linestyle="--")
        
        plt.title(f"Comparative Analysis: {title}", fontsize=13, fontweight="bold")
        plt.xlabel("Training Episodes", fontsize=11)
        plt.ylabel("Smoothed Collective System Reward", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        # Saving the figure
        fig_path = os.path.join("../figures", filename)
        plt.savefig(fig_path, dpi=150)
        print(f" Graph saved successfully to: {fig_path}")
        plt.close()

    # Exporting the dataset to CSV
    df = pd.DataFrame(all_dataset_rows)
    csv_path = os.path.join("../dataset", "simulation_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Dataset successfully exported to: {csv_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()