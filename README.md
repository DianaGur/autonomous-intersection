  Multi-Agent Reinforcement Learning for Autonomous Intersection Management

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An asymmetric stochastic game implementation comparing **WoLF-PHC** (Win or Learn Fast - Policy Hill-Climbing) and **Independent Q-Learning (IQL)** for safety-critical autonomous intersection management under normal and emergency vehicle preemption scenarios.

---

##  Project Overview

Urban intersection management represents a classic multi-agent coordination challenge where non-stationarity and safety-critical constraints must be balanced. This project evaluates how variable learning rates (WoLF-PHC) handle non-stationary environments compared to standard independent Q-learning baseline in two distinct multi-agent scenarios:

1. **Pure Scalability Scenario**: Fully symmetric multi-lane intersection with standard autonomous vehicles.
2. **Emergency Vehicle Preemption Scenario**: Asymmetric environment where dynamic priority is given to emergency vehicles (ambulances), requiring implicit yielding policies from background traffic.

---

## Repository Structure

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

## Runing instructions

    1. **Clone the repository:**
        Clone the project repository to your local computer and navigate into the root directory:
        ```bash
        git clone [https://github.com/YOUR_USERNAME/autonomous-intersection.git](https://github.com/YOUR_USERNAME/autonomous-intersection.git)
        cd autonomous-intersection
    2. **Set up virtual environment & install dependencies**:
        Create a Python 3.10 virtual environment, activate it, and install all required packages listed in requirements.txt:
        Windows (PowerShell): 
            python -m venv .venv
            .\.venv\Scripts\Activate.ps1
            pip install -r requirements.txt
        Linux / macOS:
            python3 -m venv .venv
            source .venv/bin/activate
            pip install -r requirements.txt    
    3.  **Run preliminary 2-agent experiments**
        Navigate into the src/ directory and run the preliminary experiments to evaluate basic convergence and generate the initial dataset CSV + individual reward plots:
        cd src
        python primery_experiments.py

        This step will generate output figures inside figures/ and export dataset/simulation_dataset.csv.
    4.  **Run multi-agent scalability benchmark**:
        Execute the main scalability experiment suite to benchmark performance across 4 to 16 agents:
        python main.py

        This step will update the multi-agent scalability benchmark plots inside the figures/ directory.