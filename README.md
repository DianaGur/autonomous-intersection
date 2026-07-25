  Multi-Agent Reinforcement Learning for Autonomous Intersection Management

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An asymmetric stochastic game implementation comparing **WoLF-PHC** (Win or Learn Fast - Policy Hill-Climbing) and **Independent Q-Learning (IQL)** for safety-critical autonomous intersection management under normal and emergency vehicle preemption scenarios.

---

## 📌 Project Overview

Urban intersection management represents a classic multi-agent coordination challenge where non-stationarity and safety-critical constraints must be balanced. This project evaluates how variable learning rates (WoLF-PHC) handle non-stationary environments compared to standard independent Q-learning baseline in two distinct multi-agent scenarios:

1. **Pure Scalability Scenario**: Fully symmetric multi-lane intersection with standard autonomous vehicles.
2. **Emergency Vehicle Preemption Scenario**: Asymmetric environment where dynamic priority is given to emergency vehicles (ambulances), requiring implicit yielding policies from background traffic.

---

## 🛠️ Repository Structure

```text
autonomous-intersection/
├── dataset/                        # Logged trajectory & action datasets (.csv)
│   └── simulation_dataset.csv
├── figures/                        # Output plots and comparative performance figures
│   ├── scenario_1_reward_analysis.png
│   ├── scenario_2_reward_analysis.png
│   └── multi_scenario_comparison.png
├── src/                            # Core Python source code
│   ├── agents.py                   # WoLF-PHC & Q-Learning Agent implementations
│   ├── environment.py              # MultiVehicleIntersectionEnv environment definition
│   ├── primery_experiments.py      # Script for preliminary 2-agent experiments & dataset logging
│   └── main.py                     # Main scalability experiment suite
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies