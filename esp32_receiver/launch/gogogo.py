#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import random
import tf_transformations
import os
import pickle
import csv
from datetime import datetime
import subprocess, os, random, math, xacro, time
from ament_index_python.packages import get_package_share_directory


class GoToPointRL(Node):
    def __init__(self):
        super().__init__('go_to_point_rl')

        # === ROS2 設定 ===
        self.cmd_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.create_subscription(Odometry, '/robot1/odom', self.odom_callback, 10)

        # === 強化學習參數 ===
        self.alpha = 0.1        # 學習率
        self.gamma = 0.9        # 折扣因子
        self.epsilon = 0.5      # 探索率
        self.q_table = {}       # 狀態 -> 動作分數

        # === 環境參數 ===
        self.target = (-0.5, 1)        # 固定終點
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.episode = 0
        self.prev_state = None
        self.prev_action = None
        self.episode_start_time = None # 每輪開始時間
        self.max_episode_time = 5.0   # 每輪最多 10 秒

        # 只用四個動作：前、右、左、後
        # 動作編號：0: 前進, 1: 右, 2: 左, 3: 後退
        self.actions = list(range(4))
        self.action_angles = [0, math.pi/2, -math.pi/2, math.pi]
        # self.action_angles = [0, math.pi/2, -math.pi/2]

        # === 速度設定 ===
        self.speed_linear = 0.8
        self.speed_angular = 0.4

        # === 存檔設定 ===
        data_folder = os.path.join(os.getcwd(), 'src/esp32_receiver/data')
        os.makedirs(data_folder, exist_ok=True)
        # 指定完整檔案路徑
        self.q_table_path = os.path.join(data_folder, "q_table.pkl")
        self.save_every_episodes = 1

        self.get_logger().info('Reinforcement Learning 已啟動，開始訓練！')

        # 初始重置環境
        self.reset_environment()


    # ---------------- Odometry 回調 ----------------
    def odom_callback(self, msg):
        # 更新位置與朝向
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.step_rl()


    # ---------------- 狀態離散化 ----------------
    def get_state(self):
        """將連續數值離散化成 (distance, angle_diff)"""
        tx, ty = self.target
        x, y = self.position
        dx = tx - x
        dy = ty - y
        distance = round(math.hypot(dx, dy), 1)  # 距離四捨五入到 0.1
        angle_to_goal = math.atan2(dy, dx)

        # 正規化角度差到 [-π, π]
        angle_diff = angle_to_goal - self.yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        angle_diff = round(angle_diff, 1)

        return (distance, angle_diff)


    # ---------------- 動作選擇 ----------------
    def choose_action(self, state):
        """ε-greedy 策略"""
        if random.uniform(0, 1) < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    # ---------------- Q 值更新 ----------------
    def update_q_table(self, reward, new_state):
        if self.prev_state is None or self.prev_action is None:
            return

        # 確保狀態存在
        if self.prev_state not in self.q_table:
            self.q_table[self.prev_state] = {a: 0 for a in self.actions}
        if new_state not in self.q_table:
            self.q_table[new_state] = {a: 0 for a in self.actions}

        # 取出舊 Q 值
        old_value = self.q_table[self.prev_state][self.prev_action]
        next_max = max(self.q_table[new_state].values())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.prev_state][self.prev_action] = new_value


    # ---------------- 每步執行 ----------------
    def step_rl(self):
        # === 檢查是否超時（10 秒）===
        if self.episode_start_time is not None:
            elapsed = time.time() - self.episode_start_time
            if elapsed > self.max_episode_time:
                reward = -150  # 時間懲罰
                reason = f'超時 {elapsed:.1f}s'
                self.get_logger().warn(f'Episode {self.episode} 超時！{reason}')
                self.update_q_table(reward, self.get_state())
                self.handle_episode_end(reason)
                return  # 直接結束本步

        # 取得狀態
        state = self.get_state()
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        # 選擇動作
        action = self.choose_action(state)
        twist = Twist()

        # === 根據動作設定速度 ===
        angle = self.action_angles[action]
        if abs(angle) == math.pi:  # 後退
            twist.linear.x = -self.speed_linear
            twist.angular.z = 0.0
        else:
            twist.linear.x = self.speed_linear
            twist.angular.z = self.speed_angular if angle > 0 else -self.speed_angular

        self.cmd_pub.publish(twist)
        time.sleep(0.2)
        # 停止
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)

        # === 計算基礎獎勵：越近越好 ===
        tx, ty = self.target
        x, y = self.position
        distance = math.hypot(tx - x, ty - y)
        reward = -1.0  # 每步扣 1，鼓勵快點結束
        reward += (self.prev_distance - distance) * 30  # 靠近加 30 分/米（超重要！）
        self.prev_distance = distance  # 關鍵！一定要更新！

        # 額外加分：面向目標
        angle_to_goal = math.atan2(ty - y, tx - x)
        angle_diff = abs(angle_to_goal - self.yaw)
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        if abs(angle_diff) < 0.5:  # 正負27度內
            reward += 1.0

        episode_end = False
        reason = ''

        # === 檢查超出邊界 ===
        if abs(x) > 1.2 or y < -0.7 or y > 1.1 :
            reward -= 200
            reason = f'超出範圍 ({x:.2f}, {y:.2f})'
            self.get_logger().warn(f'Episode {self.episode} 失敗！{reason}')
            episode_end = True

        # === 檢查是否到達目標 ===
        if distance < 0.25:
            reward += 150
            reason = '抵達目標'
            self.get_logger().info(f'Episode {self.episode} 完成，抵達目標！')
            episode_end = True

        # === 更新 Q 表 ===
        self.update_q_table(reward, state)
        self.prev_state = state
        self.prev_action = action

        # === 若結束，處理結算 ===
        if episode_end:
            self.handle_episode_end(reason)


    # ---------------- Episode 結束處理 ----------------
    def handle_episode_end(self, reason):
        tx, ty = self.target
        x, y = self.position
        final_distance = math.hypot(tx - x, ty - y)
        score = int(1000 / (1 + final_distance))

        elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0

        self.data_collect(reason, final_distance, elapsed, score)

        # 完整結算訊息（已修復字串問題）
        self.get_logger().info(
            f'Episode {self.episode} 結束｜'
            f'原因：{reason}｜'
            f'距離：{final_distance:.3f}｜'
            f'時間：{elapsed:.1f}s｜'
            f'分數：{score}'
        )

        # 存檔 Q-table
        try:
            self.save_q_table()
            self.get_logger().info(f'Q-table 已存檔：{self.q_table_path}')
        except Exception as e:
            self.get_logger().error(f'存檔失敗：{e}')

        # 重置環境
        self.reset_environment()


    # ---------------- 存處資料 ----------------
    def data_collect(self, reason, final_distance, elapsed, score):
        # 建立輸出資料夾
        output_folder = "results"
        os.makedirs(output_folder, exist_ok=True)

        # 設定輸出檔案（累積所有 episode 資料）
        base_path = os.path.expanduser("/home/g/ros2_ws/src/esp32_receiver/data")
        os.makedirs(base_path, exist_ok=True)
        output_file = os.path.join(base_path, "parameters.csv")

        # 準備要寫入的資料
        record = {
            "episode": self.episode,
            "reason": reason,
            "distance": round(final_distance, 3),
            "elapsed_time": round(elapsed, 1),
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 寫入 CSV（若不存在就建立）
        file_exists = os.path.isfile(output_file)
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        print(f"✅ Episode {self.episode} 資料已儲存到 {output_file}", flush=True)


    # ---------------- 重置環境 ----------------
    def reset_environment(self):
        self.episode += 1
        # self.epsilon = max(0.05, 0.3 * (0.99 ** self.episode))  # 緩慢下降
        self.reached = False
        self.prev_state = None
        self.prev_action = None

        self.get_logger().info(f'新目標：({self.target[0]:.2f}, {self.target[1]:.2f})')

        # init_x = random.uniform(-1.1, 1.0)
        # init_y = random.uniform(-0.8, 0.6)
        init_x = 0.0
        init_y = 0.0

        tx, ty = self.target
        x, y = self.position
        self.prev_distance = math.hypot(tx - x, ty - y) #計算絕對距離
        # self.prev_distance = distance
        self.position = (init_x, init_y)
        # self.yaw = random.uniform(-math.pi, math.pi)
        self.yaw = 0

        # === 停止機器人 ===
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)

        # === 刪除舊的 robot1 ===
        self.get_logger().info("刪除舊的 robot1...")
        try:
            subprocess.run([
                "ros2", "service", "call", "/delete_entity", "gazebo_msgs/srv/DeleteEntity",
                "{name: 'robot1'}"
            ], check=True)
        except Exception as e:
            self.get_logger().warn(f"刪除 robot1 失敗（可能不存在）：{e}")

        # === 展開 XACRO 成 URDF ===
        pkg_path = os.path.join(get_package_share_directory('esp32_receiver'))
        xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
        urdf_output = '/tmp/robot1.urdf'  # 暫存展開結果

        self.get_logger().info("展開 XACRO 成 URDF...")
        doc = xacro.process_file(xacro_file, mappings={'ns': 'robot1'})
        with open(urdf_output, 'w') as f:
            f.write(doc.toxml())

        # === 生成新 robot1 ===
        self.get_logger().info(f"生成新的 robot1 於 ({init_x:.2f}, {init_y:.2f}) yaw={self.yaw:.2f}")

        spawn_cmd = [
            "ros2", "run", "gazebo_ros", "spawn_entity.py",
            "-entity", "robot1",
            "-file", urdf_output,
            "-x", str(init_x),
            "-y", str(init_y),
            "-Y", str(self.yaw)
        ]

        try:
            subprocess.run(spawn_cmd, check=True)
            self.get_logger().info("新的 robot1 已生成成功。")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"生成 robot1 失敗：{e}")

        self.episode_start_time = time.time()

        # 自動存檔
        if self.episode % self.save_every_episodes == 0:
            try:
                self.save_q_table()
            except Exception as e:
                self.get_logger().error(f'自動存檔失敗：{e}')
        
        

    # ---------------- Q-table 存取 ----------------
    def save_q_table(self):
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = pickle.load(f)
            self.get_logger().info(f'已載入 Q-table：{self.q_table_path}')


# ---------------- 主程式 ----------------
def main():
    rclpy.init()
    node = GoToPointRL()

    # 嘗試載入舊的 Q-table
    try:
        node.load_q_table()
    except Exception as e:
        node.get_logger().warn(f'載入 Q-table 失敗：{e}')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('收到中斷，準備關閉...')
    finally:
        # 程式結束前強制存檔
        try:
            node.save_q_table()
            node.get_logger().info(f'程式結束，Q-table 已存檔：{node.q_table_path}')
        except Exception as e:
            node.get_logger().error(f'結束時存檔失敗：{e}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()