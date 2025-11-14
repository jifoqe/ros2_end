#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import os
import subprocess
import xacro
import time
import pickle
import random
import csv
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
import tf_transformations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


# ---------------- DQN 網路 ----------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.stack(state), np.array(action), np.array(reward),
                np.stack(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class GoToPointRL(Node):
    def __init__(self):
        super().__init__('go_to_point_rl')

        # ROS 相關
        self.cmd_pub = self.create_publisher(Twist, '/robot2/cmd_vel', 10)
        self.create_subscription(Odometry, '/robot2/odom', self.odom_callback, 10)

        # === 環境參數 ===
        self.target = (-0.5, 1.0)          # 固定終點
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.episode = 0
        self.episode_start_time = None
        self.max_episode_time = 6.0       # 每輪最多 5 秒
        self.prev_distance = 0.0
        self.prev_position = 0.0

        # === 動作空間 ===
        self.action_dim = 4
        self.actions = list(range(self.action_dim))
        self.action_angles = [0, math.pi/2, -math.pi/2, math.pi]

        # === 速度設定 ===
        self.speed_linear = 0.8
        self.speed_angular = 1.5

        # === DQN 超參數 ===
        self.state_dim = 3                # distance, angle_diff, yaw
        self.hidden = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000         # 總步數衰減
        self.batch_size = 64
        self.lr = 1e-3
        self.target_update_freq = 100      # 每多少 step 更新 target net
        self.buffer_capacity = 20000
        self.steps = 0

        # 網路與優化器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_capacity)

        # === 存檔設定 ===
        data_folder = os.path.join(os.getcwd(), 'src/esp32_receiver/data')
        os.makedirs(data_folder, exist_ok=True)
        self.model_path = os.path.join(data_folder, "dqn_model.pth")
        self.save_every_episodes = 10

        self.get_logger().info('DQN 已啟動，開始訓練！')
        self.load_model()
        self.reset_environment()

    # ---------------- 環境重置 ----------------
    def reset_environment(self):
        self.episode = self.new_episode()
        self.episode += 2
        self.get_logger().info(f'Episode {self.episode} 開始，目標：{self.target}')

        init_x = random.uniform(3.0, 5.0)
        init_y = random.uniform(-0.6, 0.6)
        # init_x = 0.0
        # init_y = 0.0
        self.yaw = random.uniform(-math.pi, math.pi)

        # 刪除舊 robot
        try:
            subprocess.run([
                "ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity",
                "{name: 'robot2'}"
            ], check=True, timeout=15)
        except Exception as e:
            self.get_logger().warn(f"刪除 robot2 失敗：{e}")

        # XACRO → URDF
        pkg_path = get_package_share_directory('esp32_receiver')
        xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
        urdf_output = '/tmp/robot2.urdf'
        doc = xacro.process_file(xacro_file, mappings={'ns': 'robot2'})
        with open(urdf_output, 'w') as f:
            f.write(doc.toxml())

        # 生成新 robot
        spawn_cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity", "robot2",
            "-file", urdf_output,
            "-x", str(init_x), "-y", str(init_y), "-Y", str(self.yaw)
        ]
        subprocess.run(spawn_cmd, check=True, timeout=10)

        self.episode_start_time = time.time()
        tx, ty = self.target
        self.prev_distance = math.hypot(tx - init_x, ty - init_y)
        self.prev_position = self.prev_distance

        # 自動存檔
        if self.episode % self.save_every_episodes == 0:
            self.save_model()

        
    def new_episode(self):
        base_path = os.path.expanduser("/home/g/ros2_ws/src/esp32_receiver/data")
        os.makedirs(base_path, exist_ok=True)
        output_file = os.path.join(base_path, "dqn_parameters.csv")

        if not os.path.isfile(output_file):
            self.get_logger().info('⚠️ 找不到 dqn_parameters.csv 檔案')
            return 0

        last_elapsed = 0
        with open(output_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_elapsed = int(row["episode"])
                except (KeyError, ValueError):
                    continue  # 跳過錯誤資料
    
        return last_elapsed


    # ---------------- 模型存取 ----------------
    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'steps': self.steps
        }, self.model_path)
        self.get_logger().info(f'DQN 模型已儲存：{self.model_path}')

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episode = checkpoint.get('episode', 0)
            self.steps = checkpoint.get('steps', 0)
            self.get_logger().info(f'已載入 DQN 模型：{self.model_path}')

    # ---------------- Odometry 回調 ----------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])
        self.step_rl()

    # ---------------- 取得狀態 (連續) ----------------
    def get_state(self):
        tx, ty = self.target
        x, y = self.position
        dx = tx - x
        dy = ty - y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        angle_diff = angle_to_goal - self.yaw
        # 正規化到 [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return np.array([distance, angle_diff, self.yaw], dtype=np.float32)

    # ---------------- ε-greedy ----------------
    def select_action(self, state):
        self.steps += 2
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps / self.epsilon_decay)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    # ---------------- DQN 更新 ----------------
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target net soft update（可選 hard update）
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------------- 每步執行 ----------------
    def step_rl(self):
        state = self.get_state()
        action = self.select_action(state)
        reward = 0
        # 檢查超時
        if self.episode_start_time is not None:
            elapsed = time.time() - self.episode_start_time
            if elapsed > self.max_episode_time:
                reason = f'超時 {elapsed:.1f}s'
                # self.handle_episode_end(reason, reward=-10)
                self.handle_episode_end(reason, final_reward=-10)
                return

        # 執行動作
        twist = Twist()
        if action == 0:      # 前進
            twist.linear.x = self.speed_linear
        elif action == 1:    # 右轉
            twist.linear.x = self.speed_linear
            twist.angular.z = self.speed_angular
        elif action == 2:    # 左轉
            twist.linear.x = self.speed_linear
            twist.angular.z = -self.speed_angular
        elif action == 3:    # 後退
            twist.linear.x = -self.speed_linear

        self.cmd_pub.publish(twist)
        time.sleep(0.2)
        self.cmd_pub.publish(Twist())  # 停止

        # 計算獎勵
        tx, ty = self.target
        x, y = self.position
        distance = math.hypot(tx - x, ty - y)
        if distance<self.prev_position:
            reward = self.prev_position - distance - 0.1 
        elif distance>self.prev_position:
            reward -= 10

        self.prev_position = distance

        # 額外獎勵
        if distance < 0.25:
            reward += 20
            reason = '抵達目標'
            done = True
        elif x < 2.8 or x > 5.2 or y < -0.7 or y > 1.1:
            reward -= 50
            reason = '超出邊界'
            done = True
        else:
            done = False
            reason = ''

        next_state = self.get_state()
        self.memory.push(state, action, reward, next_state, done)

        # 優化網路
        self.optimize_model()

        if done:
            self.handle_episode_end(reason, reward)

    # ---------------- Episode 結束 ----------------
    def handle_episode_end(self, reason, final_reward=0):
        tx, ty = self.target
        x, y = self.position
        final_distance = math.hypot(tx - x, ty - y)
        elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
        score = int(1000 / (1 + final_distance))

        self.data_collect(reason, final_distance, elapsed, score)

        self.get_logger().info(
            f'Episode {self.episode} 結束｜原因：{reason}｜'
            f'距離：{final_distance:.3f}｜時間：{elapsed:.1f}s｜分數：{score}'
        )

        # 定期存檔
        if self.episode % self.save_every_episodes == 0:
            self.save_model()

        self.reset_environment()

    # ---------------- 資料收集 ----------------
    def data_collect(self, reason, final_distance, elapsed, score):
        base_path = os.path.expanduser("/home/g/ros2_ws/src/esp32_receiver/data")
        os.makedirs(base_path, exist_ok=True)
        output_file = os.path.join(base_path, "dqn_parameters.csv")

        record = {
            "episode": self.episode,
            "reason": reason,
            "distance": round(final_distance, 3),
            "elapsed_time": round(elapsed, 1),
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        file_exists = os.path.isfile(output_file)
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)


def main():
    rclpy.init()
    node = GoToPointRL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('收到中斷，準備關閉...')
    finally:
        node.save_model()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()