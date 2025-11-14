#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt

# 載入 Q-table
with open("/home/g/ros2_ws/src/esp32_receiver/data/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# 取得每個 state 的所有 Q-values
all_q_values = [q for actions in q_table.values() for q in actions.values()]

# 取得每個 state 的最大 Q-value
max_q_per_state = [max(actions.values()) for actions in q_table.values()]

# 畫出所有 Q-values 的分布
plt.figure(figsize=(10, 5))
plt.plot(sorted(all_q_values), label="All Q-values")
plt.plot(sorted(max_q_per_state), label="Max Q-value per state", linestyle="--")
plt.title("Q-Value Distribution")
plt.xlabel("States / Actions (sorted)")
plt.ylabel("Q-Value")
plt.legend()
plt.grid(True)
plt.show()
