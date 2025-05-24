# 🧠 Deep Q-Learning Agent on Custom FrozenLake (3x3 Grid)

This is a minimalist implementation of a Deep Q-Network (DQN) agent that learns to navigate a custom-made 3x3 FrozenLake-style environment. The environment has a start state, goal state, and a hole (failure state). The agent learns to reach the goal while avoiding the hole using reinforcement learning.

---

## 📌 Features

- 🔁 **Fully Custom GridWorld**: 3x3 deterministic environment built from scratch.
- 🧠 **Deep Q-Learning**: Uses PyTorch to implement a neural network for Q-value approximation.
- 🧪 **Training Loop**: Trains the agent over multiple episodes with epsilon-greedy exploration.
- 📈 **Logging**: Displays training progress and final policy values.
- ⚡ **Lightweight**: No external environment dependencies (like OpenAI Gym).

---

## 🗺️ Environment Design

```
S - Start  
G - Goal (reward = +1)  
H - Hole (reward = -1)  
. - Safe cell (reward = 0)

Grid layout:  
[ S, ., . ]  
[ ., H, . ]  
[ ., ., G ]
```

---

## 📦 Dependencies

- Python 3.8+
- PyTorch

Install dependencies using:

```bash
pip install torch
```

---

## 🚀 How to Run

```bash
python DQN.py
```


---

## 🧠 Neural Network Architecture

- **Input size**: 9 (flattened one-hot vector of grid position)
- **Hidden layers**: 64 → 32 neurons with ReLU
- **Output**: 4 Q-values (one per action: up, down, left, right)

---
 
---

## 📌 Policy Interpretation

The policy is extracted by taking the `argmax` over the Q-values for each state. This shows what action the agent would take in each state after learning.

---

## 📚 Learning Details

| Parameter      | Value      |
|----------------|------------|
| Episodes       | 30         |
| Gamma (γ)      | 0.9        |
| Epsilon (ε)    | 1          |
| LR             | 0.01       |
| Optimizer      | Adam       |

---

## 🧪 Future Improvements

- Add generalization testing with randomized goal/hole.
- Visualize learning with matplotlib or a UI.
- Extend to larger grids (e.g., 4x4 or 5x5).
- Test performance under stochastic transitions.

---

## ✨ Output Sample (After Training)
```
Optimal Action Grid:
 →  →  ↓
 ↓  H  ↓
 →  →  G
```
---

## 🤝 Contributing

Feel free to fork and play with the environment setup or agent architecture. Open to suggestions and pull requests.

---

## 📜 License

This project is open-source under the MIT License.
