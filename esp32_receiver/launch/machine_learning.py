#!/usr/bin/env python3
#episode,reason,distance,elapsed_time,score,timestamp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import os
import subprocess, threading
import xacro
import time
import pickle
import random
import csv
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
import tf_transformations
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


def safe_delete_entity(robot_name, timeout=30):
    """安全刪除 Gazebo 實體，避免 subprocess 卡死"""
    cmd = [
        "ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity",
        f'{{name: "{robot_name}"}}'
    ]

    try:
        # 用 Popen 取代 run，可控性更高
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start = time.time()

        # 每 0.5 秒檢查是否結束
        while proc.poll() is None:
            if time.time() - start > timeout:
                proc.terminate()  # 嘗試中止
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()  # 強制殺掉
                print(f"⚠️ 刪除 {robot_name} 超時，已強制終止 subprocess")
                return False
            time.sleep(0.5)

        if proc.returncode == 0:
            print(f"✅ 成功刪除 {robot_name}")
            return True
        else:
            print(f"❌ 刪除 {robot_name} 失敗，返回碼 {proc.returncode}")
            return False

    except Exception as e:
        print(f"❌ 刪除 {robot_name} 例外錯誤：{e}")
        return False
    

def spawn_with_timeout(cmd, timeout=60):
    """安全生成 robot（防止 Gazebo 卡住）"""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        timer = threading.Timer(timeout, lambda: proc.kill())
        timer.start()
        proc.wait()
        timer.cancel()

        if proc.returncode == 0:
            print(f"✅ 成功生成 {cmd[-8]}")  # 顯示 robot 名稱
            return True
        else:
            print(f"⚠️ spawn_entity 返回異常 returncode={proc.returncode}")
            return False

    except Exception as e:
        print(f"❌ spawn_entity 發生錯誤：{e}")
        return False

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
    def __init__(self, robot_name):
        super().__init__('go_to_point_rl')
        self.robot_name = robot_name
        self.robot_index = int(self.robot_name[-1])-1

        # ROS 相關
        self.cmd_pub = self.create_publisher(Twist, f'/{robot_name}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/{robot_name}/odom', self.odom_callback, 10)

        # === 環境參數 ===
        self.target = [(-0.5, 1.0), (3.5, 1.0), (7.5, 1.0),
                       (-0.5, -3.0), (3.5, -3.0), (7.5, -3.0),
                       (-0.5, -7.0), (3.5, -7.0), (7.5, -7.0)]       # 固定終點
        self.position = (0.0, 0.0)
        self.position_new_x = [(-1, 1),(3, 5),(7, 9),
                               (-1, 1),(3, 5),(7, 9),
                               (-1, 1),(3, 5),(7, 9),]
        self.position_new_y = [(-0.6, 0.6),(-0.6, 0.6),(-0.6, 0.6),
                                (-4.6, -3.4),(-4.6, -3.4),(-4.6, -3.4),
                                (-8.6, -7.4),(-8.6, -7.4),(-8.6, -7.4),]
        self.over_x = [(-1.2, 1.2),(2.8, 5.2),(6.8, 9.2),
                       (-1.2, 1.2),(2.8, 5.2),(6.8, 9.2),
                       (-1.2, 1.2),(2.8, 5.2),(6.8, 9.2)]
        self.over_y = [(-0.7, 1.1),(-0.7, 1.1),(-0.7, 1.1),
                       (-4.7, -2.9),(-4.7, -2.9),(-4.7, -2.9),
                       (-8.7, -6.9),(-8.7, -6.9),(-8.7, -6.9)]
        self.yaw = 0.0
        self.episode = int(self.robot_index)-5
        self.episode_start_time = None
        self.max_episode_time = 7.0       # 每輪最多 5 秒
        self.prev_distance = 0.0
        self.prev_position = 0.0
        self.control_rate = self.create_rate(10)

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

        # self.get_logger().info('DQN 已啟動，開始訓練！')
        self.load_model()
        self.reset_environment()

    # ---------------- 環境重置 ----------------
    def reset_environment(self):
        self.episode += 9
        self.get_logger().info(f'[{self.robot_name}] Episode {self.episode} 開始，目標：{self.target[self.robot_index]}')

        x_min, x_max = self.position_new_x[self.robot_index]
        y_min, y_max = self.position_new_y[self.robot_index]
        init_x = random.uniform(x_min, x_max)
        init_y = random.uniform(y_min, y_max)
        # init_y = random.uniform(-0.6, 0.6)
        # init_x = 0.0
        # init_y = 0.0
        self.yaw = random.uniform(-math.pi, math.pi)

        # 刪除舊 robot
        # try:
        #     subprocess.run([
        #         "ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity",
        #         f'{{name: "{self.robot_name}"}}'], 
        #         stdout=subprocess.DEVNULL,
        #         stderr=subprocess.DEVNULL,
        #         check=True, 
        #         timeout=120
        #     )       
        # except Exception as e:
        #     self.get_logger().warn(f"刪除 {self.robot_name} 失敗：{e}")

        # 刪除舊 robot（安全版）
        if not safe_delete_entity(self.robot_name, timeout=30):
            self.get_logger().warn(f"刪除 {self.robot_name} 超時或失敗，將繼續嘗試生成新模型")

        time.sleep(1.0)

        # XACRO → URDF
        pkg_path = get_package_share_directory('esp32_receiver')
        xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
        urdf_output = f'/tmp/{self.robot_name}.urdf'
        doc = xacro.process_file(xacro_file, mappings={'ns': f'{self.robot_name}'})
        with open(urdf_output, 'w') as f:
            f.write(doc.toxml())

        # 生成新 robot
        spawn_cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity", f"{self.robot_name}",
            "-file", urdf_output,
            "-x", str(init_x), "-y", str(init_y), "-Y", str(self.yaw)
        ]
        # subprocess.run(
        #     spawn_cmd, 
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     check=True, 
        #     timeout=120
        # )
        if not spawn_with_timeout(spawn_cmd, timeout=60):
            self.get_logger().warn(f"{self.robot_name} 生成失敗，跳過此回合")
            return

        self.episode_start_time = time.time()
        tx, ty = self.target[self.robot_index]
        self.prev_distance = math.hypot(tx - init_x, ty - init_y)
        self.prev_position = self.prev_distance

        # 自動存檔
        if self.episode % self.save_every_episodes == 0:
            self.save_model()

    # ---------------- 模型存取 ----------------
    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'steps': self.steps
        }, self.model_path)
        # self.get_logger().info(f'DQN 模型已儲存：{self.model_path}')

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.episode = checkpoint.get('episode', 0)
            self.steps = checkpoint.get('steps', 0)
            # self.get_logger().info(f'已載入 DQN 模型：{self.model_path}')

    # ---------------- Odometry 回調 ----------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])
        self.step_rl()

    # ---------------- 取得狀態 (連續) ----------------
    def get_state(self):
        tx, ty = self.target[self.robot_index]
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
        self.steps += 1
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
                # reason = f'超時 {elapsed:.1f}s'
                # self.handle_episode_end(reason, final_reward=-10)
                reward = -10
                next_state = self.get_state()
                self.memory.push(state, action, reward, next_state, True)
                self.handle_episode_end("超時", reward)
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
        # self.control_rate.sleep()
        time.sleep(0.2)
        self.cmd_pub.publish(Twist())  # 停止

        # 計算獎勵
        tx, ty = self.target[self.robot_index]
        x, y = self.position
        distance = math.hypot(tx - x, ty - y)
        # reward = self.prev_position - distance - 0.1   # 越靠近越好
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
        elif x < self.over_x[self.robot_index][0] \
            or x > self.over_x[self.robot_index][1] \
            or y < self.over_y[self.robot_index][0] \
            or y > self.over_y[self.robot_index][1]:
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
        tx, ty = self.target[self.robot_index]
        x, y = self.position
        final_distance = math.hypot(tx - x, ty - y)
        elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
        score = int(1000 / (1 + final_distance))

        self.data_collect(reason, final_distance, elapsed, score)

        # self.get_logger().info(
        #     f'Episode {self.episode} 結束｜原因：{reason}｜'
        #     f'距離：{final_distance:.3f}｜時間：{elapsed:.1f}s｜分數：{score}'
        # )

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
    node = GoToPointRL(robot_name=sys.argv[1])
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