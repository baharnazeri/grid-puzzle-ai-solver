# 🧠 Grid Puzzle AI Solver (ShoverWorld)

An AI agent designed to solve a custom grid-based puzzle environment using a **Weighted A\*** search algorithm.  
The project combines **heuristic search, environment simulation (Gym-style), and interactive visualization (Pygame)**.

---

## 🚀 Features

- Custom grid-based puzzle environment  
- Weighted A* search with heuristic optimization  
- Efficient state representation and action pruning  
- Fast environment cloning for scalable search  
- Interactive GUI using Pygame  
- Support for custom challenge maps  

---

## 🧩 Environment Overview

The environment is a grid world where:

- `B` → Box (movable object)  
- `L` → Lava (destroys boxes)  
- `#` / `X` → Obstacles / barriers  
- `.` → Empty space  

**Goal:**  
Move and eliminate all boxes using strategic pushes and special actions.

---

## 🤖 AI Approach

The solver is based on **Weighted A\*** search:

- Heuristic: Distance of boxes to nearest boundary  
- State representation: Grid + quantized stamina  
- Optimization techniques:
  - Action pruning  
  - Efficient state encoding  
  - Open-list pruning for scalability  

---

## 🗺️ Example Map

```txt
LLLLLLLLLLLLL
L..XXX.XXX..L
L..BB.X.X..BL
L.XXX.X.X.XXL
L..BB.L..B..L
L.XX.X.X.XXXL
L..BB.X.X..BL
L..XXX.XXX..L
LLLLLLLLLLLLL
```

---

## 🖥️ Installation

```bash
git clone https://github.com/your-username/grid-puzzle-ai-solver.git
cd grid-puzzle-ai-solver
pip install numpy pygame gym
```

---

## ▶️ Usage

### Run AI Solver

```bash
python run_ai.py
```

- Loads a challenge map  
- Runs Weighted A* search  
- Outputs solution and performance  

---

### Run GUI (Interactive Mode)

```bash
python run_gui.py
```

Or load a specific map:

```bash
python run_gui.py maps/challenge_03.txt
```

---

## 🎮 Controls (GUI)

- Click on a box → select it  
- Arrow keys / WASD → move  
- `B` → Barrier Maker  
- `H` → Hellify  
- `R` → Reset  
- `Q` → Quit  

---

## 📁 Project Structure

```
.
├── environment.py    # Core environment logic
├── player_ai.py      # Weighted A* solver
├── gui.py            # Pygame visualization
├── run_ai.py         # AI execution script
├── run_gui.py        # GUI launcher
├── utils.py          # Action encoding/decoding
├── challenge_03.txt  # Example map
└── README.md
```

---

## 📊 Key Concepts

- Heuristic Search (A*)  
- State-Space Optimization  
- Grid-Based Planning  
- Simulation Environments  
- Game AI  

---

## 🎯 Future Improvements

- Reinforcement Learning agents (DQN / PPO)  
- Learned heuristics  
- Multi-agent extensions  
- Performance benchmarking  

