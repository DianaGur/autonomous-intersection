import numpy as np

class WolfAgent:
    def __init__(self, alpha=0.15, delta_win=0.02, delta_lose=0.08):
        self.alpha, self.delta_win, self.delta_lose = alpha, delta_win, delta_lose
        self.q_table, self.policy, self.avg_policy, self.counter = {}, {}, {}, {}

    def _init_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(2)
            self.policy[s] = np.array([0.5, 0.5])
            self.avg_policy[s] = np.array([0.5, 0.5])
            self.counter[s] = 0

    def get_action(self, s, eps):
        self._init_state(s)
        if np.random.rand() < eps:
            return np.random.choice([0, 1])
        return 1 if np.random.rand() < self.policy[s][1] else 0

    def learn(self, s, a, r, ns):
        self._init_state(s); self._init_state(ns)
        mq = np.max(self.q_table[ns])
        self.q_table[s][a] += self.alpha * (r + 0.9 * mq - self.q_table[s][a])
        self.counter[s] += 1
        self.avg_policy[s] += (self.policy[s] - self.avg_policy[s]) / self.counter[s]
        cur = np.sum(self.policy[s] * self.q_table[s])
        avg = np.sum(self.avg_policy[s] * self.q_table[s])
        delta = self.delta_win if cur > avg else self.delta_lose
        best = np.argmax(self.q_table[s]); other = 1 - best
        shift = min(delta, self.policy[s][other])
        self.policy[s][best] += shift
        self.policy[s][other] -= shift
        self.policy[s] = np.clip(self.policy[s], 0, 1)
        self.policy[s] /= np.sum(self.policy[s])


class QLearningAgent:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.q_table = {}

    def _init_state(self, s):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(2)

    def get_action(self, s, eps):
        self._init_state(s)
        if np.random.rand() < eps:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[s])

    def learn(self, s, a, r, ns):
        self._init_state(s); self._init_state(ns)
        mq = np.max(self.q_table[ns])
        self.q_table[s][a] += self.alpha * (r + 0.9 * mq - self.q_table[s][a])