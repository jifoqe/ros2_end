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

def safe_delete_entity(robot_name, max_retries=3, timeout=10):
    cmd = ["ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity",
           f'{{name: "{robot_name}"}}']
    
    for attempt in range(max_retries):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                outs, errs = proc.communicate(timeout=timeout)
                if proc.returncode == 0:
                    print(f"✅ 成功刪除 {robot_name}")
                    return True
                else:
                    print(f"⚠️ 刪除失敗 (嘗試 {attempt+1}): {errs.decode()}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"⏰ 刪除超時 (嘗試 {attempt+1})")
        except Exception as e:
            print(f"❌ 刪除異常: {e}")

        time.sleep(2)  # 等待 Gazebo 穩定

    print(f"❌ 最終刪除 {robot_name} 失敗")
    return False
    
def spawn_with_timeout(cmd, timeout=10):
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
        # self.episode = int(self.robot_index)-8
        self.episode = int(self.robot_index)-8
        self.episode_start_time = None
        self.max_episode_time = 40.0       # 每輪最多 5 秒
        self.prev_distance = 0.0
        self.prev_position = 0.0
        # self.control_rate = self.create_rate(10)

        # === 動作空間 ===
        self.action_dim = 8
        self.walk_number = 0
        # self.actions = list(range(self.action_dim))
        # self.action_angles = [0, math.pi/2, -math.pi/2, math.pi]
        self.ready_for_step = True

        # === 速度設定 ===
        self.speed_linear = 0.1
        self.speed_angular = 0.1

        # === DQN 超參數 ===
        self.state_dim = 3                # x ,y , yaw
        self.hidden = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.eps_phase1_steps = 2000
        self.eps_phase2_steps = 5000
        # self.epsilon_decay = 5000         # 總步數衰減
        self.batch_size = 64
        self.lr = 1e-4
        self.target_update_freq = 100      # 每多少 step 更新 target net
        self.buffer_capacity = 20000
        self.steps = 0

        # === 移動參數 ===
        self.target_gx = 0
        self.target_gy = 0
        self.action = None
        self.state = None
        self.reset = True

        # 網路與優化器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_capacity)

        # === 存檔設定 ===
        data_folder = os.path.join(os.getcwd(), 'src/ros2_end/esp32_receiver/data')
        os.makedirs(data_folder, exist_ok=True)
        self.model_path = os.path.join(data_folder, "dqn_model.pth")
        self.save_every_episodes = 10

        # self.get_logger().info('DQN 已啟動，開始訓練！')
        self.load_model()
        self.reset_environment()

    # ---------------- 環境重置 ----------------
    def reset_environment(self):
        self.episode += 9
        self.walk_number = 0
        self.ready_for_step = True
        self.reset = True
        # self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        # self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        self.get_logger().info(f'[{self.robot_name}] Episode {self.episode} 開始，目標：{self.target[self.robot_index]}')

        x_min, x_max = self.position_new_x[self.robot_index]
        y_min, y_max = self.position_new_y[self.robot_index]
        self.init_x  = random.uniform(x_min, x_max)
        self.init_y = random.uniform(y_min, y_max)
        # self.yaw = random.uniform(-math.pi, math.pi)
        # init_y = random.uniform(-0.6, 0.6)
        # self.init_x = 0.0
        # self.init_y = 0.0
        self.yaw = 0

        # 刪除舊 robot（安全版）
        if not safe_delete_entity(self.robot_name, timeout=20):
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
            "-x", str(self.init_x), "-y", str(self.init_y), "-Y", str(self.yaw)
        ]

        if not spawn_with_timeout(spawn_cmd, timeout=20):
            self.get_logger().warn(f"{self.robot_name} 生成失敗，跳過此回合")
            return
        
        self.episode_start_time = time.time()
        tx, ty = self.target[self.robot_index]
        self.prev_distance = math.hypot(tx - self.init_x, ty - self.init_y)
        self.prev_position = self.prev_distance

        # 自動存檔
        if self.episode % self.save_every_episodes == 0:
            self.save_model()

        # self.step_rl()

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
            # self.steps = checkpoint.get('steps', 0)
            # self.get_logger().info(f'已載入 DQN 模型：{self.model_path}')

    # ---------------- Odometry 回調 ----------------
    def odom_callback(self, msg):
        # elapsed = time.time() - self.episode_start_time
        # temp_x = msg.pose.pose.position.x
        # temp_y = msg.pose.pose.position.y
        
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        x, y = self.position
        # if elapsed<0.3:
        #     self.get_logger().info("Odom time, starting control…")
        #     # self.get_logger().info(f'釋放  x:{temp_x}   y:{temp_y} 時間：{elapsed}')
        #     return
        
        dx = abs(self.init_x -x)
        dy = abs(self.init_y -y)
        if self.reset:
            if dx > 0.01 or dy > 0.01:
                f"[跳動] 位置仍不穩定 dx={dx:.4f} dy={dy:.4f}"
                return
            else:
                self.reset = False

        # self.position = (temp_x, temp_y)
        # self.get_logger().info(f'odom  x:{x}   y:{y}')
        self.step_rl()
    
    def get_state(self, reference_position=(0.0,0.0), reference_yaw=0.0):
        """
            將當前 robot 的狀態轉換到 reference_position 與 reference_yaw 座標系
            reference_position: 參考點，例如 robot1 的 (x1, y1)
            reference_yaw: 參考點朝向，例如 robot1 的 yaw
        """
        # 取得 robot 當前世界座標
        x, y = self.position
        # self.get_logger().info(f'真實座標  x:{x}   y:{y}')
        # tx, ty = self.target[self.robot_index]

        # 平移到 reference 座標系
        x_rel = x - reference_position[0]
        y_rel = y - reference_position[1]
        # 旋轉到 reference yaw
        # cos_r = math.cos(-reference_yaw)
        # sin_r = math.sin(-reference_yaw)
        # x_rel = dx * cos_r - dy * sin_r
        # y_rel = dx * sin_r + dy * cos_r

        # 目標也轉到同樣座標系
        # dx_target = tx - reference_position[0]
        # dy_target = ty - reference_position[1]
        # tx_rel = dx_target * cos_r - dy_target * sin_r
        # ty_rel = dx_target * sin_r + dy_target * cos_r


        # 4) 區間量化（每 5 單位一格，向下/更負方向取整）
        grid = 0.25
        self.x_q = math.floor(x_rel / grid) * grid
        self.y_q = math.floor(y_rel / grid) * grid

        # 計算距離和角度
        # distance = math.hypot(tx_rel - x_rel, ty_rel - y_rel)
        # angle_to_goal = math.atan2(ty_rel - y_rel, tx_rel - x_rel)
        # angle_diff = angle_to_goal - 0.0  # 在 reference 座標系下，reference yaw 已經當作 0
        # 正規化 [-π, π]
        # while angle_diff > math.pi:
        #     angle_diff -= 2 * math.pi
        # while angle_diff < -math.pi:
        #     angle_diff += 2 * math.pi

        # yaw 可選，轉換到 reference 座標系
        yaw_rel = self.yaw - reference_yaw
        while yaw_rel > math.pi:
            yaw_rel -= 2 * math.pi
        while yaw_rel < -math.pi:
            yaw_rel += 2 * math.pi

        # -----------------
        # 🔥 正規化（非常重要）
        # -----------------
        # distance_norm   = distance / 2.5          # 建議 2.5 ~ 5 視你的地圖而定
        # angle_diff_norm = angle_diff / math.pi
        # yaw_norm        = yaw_rel / math.pi
        # # 限制 [-1, 1]
        # distance_norm = max(-1, min(1, distance_norm))
        # return np.array([distance_norm, angle_diff_norm, yaw_norm], dtype=np.float32)
        return np.array([self.x_q, self.y_q, yaw_rel], dtype=np.float32)


    def get_epsilon(self):
        if self.episode <= self.eps_phase1_steps:
            # phase1: 1.0 -> 0.2
            return 1.0 - (1.0 - 0.2) * (self.episode / self.eps_phase1_steps)
        elif self.episode <= self.eps_phase2_steps:
            # phase2: 0.2 -> 0.05
            return 0.2 - (0.2 - self.epsilon_end) * ((self.episode - self.eps_phase1_steps) / (self.eps_phase2_steps - self.eps_phase1_steps))
        else:
            return self.epsilon_end

    # ---------------- ε-greedy ----------------
    def select_action(self, state):
        # self.steps += 1
        epsilon = self.get_epsilon()
        # epsilon = 1.0 - self.steps*0.01
        self.get_logger().info(f'epsilon:{epsilon}')
        # epsilon = 0.9
        # epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #           math.exp(-1. * self.steps / self.epsilon_decay)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

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

        # next_q = self.target_net(next_states).max(1)[0].detach()
        # target_q = rewards + self.gamma * next_q * (1 - dones)

        # === Double DQN 核心修改開始 ===
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1).detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        # === Double DQN 修改結束 ===

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target net soft update（可選 hard update）
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------------- 每步執行 ----------------
    def step_rl(self):
        grid_size = 0.25
        
        # 目標格子座標
        if self.ready_for_step == True:
            self.state = self.get_state()
            gx = self.x_q
            gy = self.y_q
            self.action = self.select_action(self.state)
            action_offsets = [
                (-1,  1), (0,  1), (1,  1),
                (-1,  0),          (1,  0),
                (-1, -1), (0, -1), (1, -1)
            ]
            dx, dy = action_offsets[self.action]
            dx, dy = action_offsets[self.action]
            self.target_gx = gx + dx * grid_size
            self.target_gy = gy + dy * grid_size
            self.get_logger().info(f'x:{gx}   y:{gy}')
            self.get_logger().info(f'x:{dx}   y:{dy}')
            self.get_logger().info(f'x:{self.target_gx}   y:{self.target_gy}')
        
        elapsed = time.time() - self.episode_start_time
        reward = 0

        if self.episode_start_time is not None:
            if elapsed > self.max_episode_time:
                reward = -20
                next_state = self.get_state()
                self.memory.push(self.state, self.action, reward, next_state, True)
                self.handle_episode_end("超時", reward)
                return
    
        x, y = self.position
        dx_move = self.target_gx - x
        dy_move = self.target_gy - y
        distance = math.hypot(dx_move, dy_move)
        # 計算局部座標下的速度（考慮車子 yaw 方向）
        # target_angle = math.atan2(dy_move, dx_move)
        # angle_diff = target_angle - self.yaw
        # cos_yaw = math.cos(self.yaw)
        # sin_yaw = math.sin(self.yaw)
        # local_dx = cos_yaw * dx_move + sin_yaw * dy_move
        # local_dy = -sin_yaw * dx_move + cos_yaw * dy_move
        # while angle_diff > math.pi: angle_diff -= 2*math.pi
        # while angle_diff < -math.pi: angle_diff += 2*math.pi
        twist = Twist()

        # 限制速度
        if distance > 0.05: 
            # self.get_logger().info(f'現在的情況：位置:{x:.2f}  {y:.2f}')
            twist.linear.x = max(-self.speed_linear, min(self.speed_linear, dx_move))
            twist.linear.y = max(-self.speed_linear, min(self.speed_linear, dy_move))
            # twist.angular.z = max(-self.speed_angular, min(self.speed_angular, 2*angle_diff))
            self.cmd_pub.publish(twist)
            self.ready_for_step = False
            return
        
        # if distance > 0.05:  # 還沒到目標格子中心
        #     # self.get_logger().info(f'現在的情況：{angle_diff}  位置:{x}  {y}')
        #     if abs(angle_diff) > 0.2:
        #         # 角度偏差大 → 只轉向
        #         twist.linear.x = 0.0
        #         twist.angular.z = max(-self.speed_angular, min(self.speed_angular, 2*angle_diff))
        #         # twist.angular.z = self.speed_angular
        #     else:
        #         # 角度對準 → 前進
        #         twist.linear.x = self.speed_linear
        #         twist.angular.z = max(-self.speed_angular, min(self.speed_angular, 2*angle_diff))
        #         # twist.angular.z = self.speed_angular
        #     self.cmd_pub.publish(twist)
        #     self.ready_for_step = False
        #     return
        
        self.ready_for_step = True
        self.get_logger().info(f'完成：到達格子中心 ({x:.2f}, {y:.2f})')
        self.walk_number += 1

        # 到達格子中心，計算 reward
        # reward = getattr(self, 'prev_distance', distance) - distance
        # self.prev_distance = distance
        # self.prev_grid = current_grid
        tx, ty = self.target[self.robot_index]
        distance_to_goal = math.hypot(tx - x, ty - y)

        # 距離終點越遠 → 分數越負
        reward = -distance_to_goal*10

        # 額外獎勵 / done 判斷
        # tx, ty = self.target[self.robot_index]
        if distance_to_goal := math.hypot(tx - x, ty - y) < 0.25:
            reward += 100
            done = True
            reason = '抵達目標'
        elif x < self.over_x[self.robot_index][0] or x > self.over_x[self.robot_index][1] \
            or y < self.over_y[self.robot_index][0] or y > self.over_y[self.robot_index][1]:
            reward -= 50
            done = True
            reason = '超出邊界'
        else:
            done = False
            reason = ''

        next_state = self.get_state()
        self.memory.push(self.state, self.action, reward, next_state, done)
        self.optimize_model()

        if done:
            self.handle_episode_end(reason, reward)

    # ---------------- Episode 結束 ----------------
    def handle_episode_end(self, reason, final_reward=0):
        tx, ty = self.target[self.robot_index]
        x, y = self.position
        final_distance = math.hypot(tx - x, ty - y)
        elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0

        self.data_collect(reason, final_distance, elapsed, final_reward)

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
        base_path = os.path.expanduser("/home/u/ros2_ws/src/ros2_end/esp32_receiver/data")
        os.makedirs(base_path, exist_ok=True)
        output_file = os.path.join(base_path, "dqn_parameters.csv")

        record = {
            "episode": self.episode,
            "reason": reason,
            "distance": round(final_distance, 3),
            "elapsed_time": round(elapsed, 1),
            "score": score,
            "walk_number": self.walk_number,
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