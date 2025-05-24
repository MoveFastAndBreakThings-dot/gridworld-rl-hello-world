import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Environment setup
grid_size = 3
n_states = grid_size * grid_size
n_actions = 4  # 0: up, 1: right, 2: down, 3: left

start_state = 0
goal_state = 8
hole_state = 4
action_symbols = ['↑', '→', '↓', '←']

def get_next_state(state: int, action: int) -> int:
    row, col = divmod(state, grid_size)
    if action == 0 and row > 0: row -= 1
    elif action == 1 and col < grid_size - 1: col += 1
    elif action == 2 and row < grid_size - 1: row += 1
    elif action == 3 and col > 0: col -= 1
    return row * grid_size + col

def get_reward(state: int) -> int:
    if state == goal_state:
        return 1
    elif state == hole_state:
        return -1
    else:
        return 0

def is_terminal(state: int) -> bool:
    return state == goal_state or state == hole_state

# DQN network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks(np.arange(0, grid_size + 1, 1))
ax.set_yticks(np.arange(0, grid_size + 1, 1))
ax.grid(True)
plt.gca().invert_yaxis()

for i in range(n_states):
    row, col = divmod(i, grid_size)
    color = 'white'
    if i == start_state: color = 'lightblue'
    elif i == goal_state: color = 'lightgreen'
    elif i == hole_state: color = 'red'
    ax.add_patch(patches.Rectangle((col, row), 1, 1, edgecolor='black', facecolor=color))
    ax.text(col + 0.5, row + 0.5, str(i), ha='center', va='center', fontsize=12)

agent_circle = patches.Circle((0.5, 0.5), 0.25, color='orange')
ax.add_patch(agent_circle)
title = ax.set_title("")
plt.ion()
plt.show()

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

buffer = ReplayBuffer()
batch_size = 32
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.95
min_epsilon = 0.1
episodes = 30
max_steps = 50
target_update_freq = 5

def state_to_tensor(state):
    one_hot = np.zeros(n_states)
    one_hot[state] = 1
    return torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(device)

# Training loop
for ep in range(1, episodes + 1):
    state = start_state
    done = False
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    for step in range(max_steps):
        state_tensor = state_to_tensor(state)

        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        done = is_terminal(next_state)

        buffer.add((state, action, reward, next_state, done))
        state = next_state

        # Move agent
        agent_row, agent_col = divmod(state, grid_size)
        agent_circle.center = (agent_col + 0.5, agent_row + 0.5)
        title.set_text(f"Episode {ep} | Step {step} | State {state} | Epsilon {round(epsilon, 2)} | Reward {reward}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.2)

        # Learn from batch
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            state_batch = torch.cat([state_to_tensor(s) for s in states])
            next_state_batch = torch.cat([state_to_tensor(s) for s in next_states])
            action_batch = torch.tensor(actions).unsqueeze(1).to(device)
            reward_batch = torch.tensor(rewards).float().unsqueeze(1).to(device)
            done_batch = torch.tensor(dones).float().unsqueeze(1).to(device)

            q_values = policy_net(state_batch).gather(1, action_batch)
            next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
            expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

            loss = loss_fn(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    if ep % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

plt.ioff()
plt.show()

# Final policy display
print("\nOptimal Action Grid:")
for row in range(grid_size):
    row_actions = []
    for col in range(grid_size):
        state = row * grid_size + col
        if state == goal_state:
            row_actions.append(" G ")
        elif state == hole_state:
            row_actions.append(" H ")
        else:
            with torch.no_grad():
                s_tensor = state_to_tensor(state)
                best_action = policy_net(s_tensor).argmax().item()
            row_actions.append(f" {action_symbols[best_action]} ")
    print("".join(row_actions))
