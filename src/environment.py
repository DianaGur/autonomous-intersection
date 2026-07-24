import numpy as np

LANES = ["N", "S", "E", "W"]
AXIS = {"N": "vertical", "S": "vertical", "E": "horizontal", "W": "horizontal"}

class MultiVehicleIntersectionEnv:
    def __init__(self, queue_depth, mode="pure"):
        assert mode in ("pure", "ambulance_random")
        self.mode = mode
        self.q = queue_depth
        self.intersection_idx = self.q
        self.goal_idx = self.q + 1

        self.agent_lane = {}
        aid = 0
        for lane in LANES:
            for slot in range(self.q):
                self.agent_lane[aid] = lane
                aid += 1
        self.n_agents = aid

        # front-of-queue agent id for each lane (used to pick the ambulance)
        self.lane_front_agent = {
            lane: max(i for i in self.agent_lane if self.agent_lane[i] == lane)
            for lane in LANES
        }
        self.reset()

    def reset(self):
        self.pos = {}
        for lane in LANES:
            ids = [i for i in self.agent_lane if self.agent_lane[i] == lane]
            for rank, aid in enumerate(ids):
                self.pos[aid] = rank
        self.done = {i: False for i in range(self.n_agents)}
        self.crash = False

        if self.mode == "ambulance_random":
            self.ambulance_lane_idx = np.random.randint(4)
            self.ambulance_lane = LANES[self.ambulance_lane_idx]
            self.ambulance_id = self.lane_front_agent[self.ambulance_lane]
        else:
            self.ambulance_lane_idx = -1
            self.ambulance_id = None

        return self._get_state()

    def _get_state(self):
        base = tuple(self.pos[i] for i in range(self.n_agents))
        if self.mode == "ambulance_random":
            return base + (self.ambulance_lane_idx,)
        return base

    def step(self, actions):
        rewards = {i: 0 for i in range(self.n_agents)}
        if self.crash:
            return self._get_state(), {i: -50 for i in range(self.n_agents)}, True

        prev_pos = dict(self.pos)
        order = sorted(range(self.n_agents), key=lambda i: -self.pos[i])
        occupied = {self.pos[i] for i in range(self.n_agents) if not self.done[i]}

        for i in order:
            if self.done[i]:
                continue
            if actions[i] == 1:
                target = self.pos[i] + 1
                lane = self.agent_lane[i]
                blocked = any(
                    (not self.done[j]) and self.agent_lane[j] == lane and self.pos[j] == target
                    for j in range(self.n_agents) if j != i
                )
                if blocked:
                    rewards[i] -= 0.5
                    continue
                occupied.discard(self.pos[i])
                self.pos[i] = target
                occupied.add(target)
                if self.pos[i] == self.goal_idx:
                    self.done[i] = True
                    rewards[i] += 20
                    occupied.discard(target)
            else:
                rewards[i] -= 0.5

        at_box = [i for i in range(self.n_agents) if not self.done[i] and self.pos[i] == self.intersection_idx]
        for a in range(len(at_box)):
            for b in range(a + 1, len(at_box)):
                i, j = at_box[a], at_box[b]
                if AXIS[self.agent_lane[i]] != AXIS[self.agent_lane[j]]:
                    self.crash = True
                    rewards[i] -= 100
                    rewards[j] -= 100

        if self.ambulance_id is not None and not self.done[self.ambulance_id]:
            amb_prev = prev_pos[self.ambulance_id]
            amb_axis = AXIS[self.agent_lane[self.ambulance_id]]
            if amb_prev == self.intersection_idx - 1:
                for i in range(self.n_agents):
                    if i == self.ambulance_id or self.done[i]:
                        continue
                    if AXIS[self.agent_lane[i]] == amb_axis:
                        continue
                    if self.pos[i] == self.intersection_idx:
                        rewards[i] -= 40
                    elif prev_pos[i] == self.intersection_idx - 1 and self.pos[i] == prev_pos[i]:
                        rewards[i] += 8

        done = all(self.done.values()) or self.crash
        return self._get_state(), rewards, done
