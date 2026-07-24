import numpy as np
from environment import MultiVehicleIntersectionEnv
from agents import WolfAgent, QLearningAgent

def run_experiment(agent_type, queue_depth, episodes, seed, mode):
    np.random.seed(seed)
    env = MultiVehicleIntersectionEnv(queue_depth, mode=mode)
    n = env.n_agents
    agents = {i: (WolfAgent() if agent_type == "wolf" else QLearningAgent()) for i in range(n)}
    max_steps = 4 * (queue_depth + 2)

    crash_hist, success_hist = [], []
    for ep in range(episodes):
        state = env.reset()
        steps = 0
        while not all(env.done.values()) and not env.crash and steps < max_steps:
            epsilon = max(0.02, 0.3 * (1 - ep / (episodes * 0.7)))
            actions = {i: agents[i].get_action(state, epsilon) for i in range(n)}
            next_state, rewards, done = env.step(actions)
            for i in range(n):
                agents[i].learn(state, actions[i], rewards[i], next_state)
            state = next_state
            steps += 1
        crash_hist.append(1 if env.crash else 0)
        success_hist.append(1 if all(env.done.values()) and not env.crash else 0)
    return crash_hist, success_hist


def last_window_mean(vals, n=200):
    return float(np.mean(vals[-n:]))
