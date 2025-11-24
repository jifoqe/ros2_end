#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf_transformations

class GoToPoint(Node):
    def __init__(self):
        super().__init__('go_to_point')

        # ROS Publisher / Subscriber
        self.cmd_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.create_subscription(Odometry, '/robot1/odom', self.odom_callback, 10)

        # 目標座標
        self.target = (-1, -0.5)  # 想要到達的座標
        self.position = (0.0, 0.0)
        self.yaw = 0.0

        # 控制參數
        self.max_linear_speed = 0.05    # 最大線速度
        self.distance_tolerance = 0.05  # 到目標的容忍距離

        self.reached_target = False     # 用來判斷是否已到達目標

    def odom_callback(self, msg):
        # 更新位置與朝向
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        _, _, self.yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        local_dx , local_dy = self.position

        # 每次收到 odom 就更新控制
        # self.get_logger().info(f"出發")
        self.get_logger().info(f"local_dx: {local_dx:.3f}, local_dy: {local_dy:.3f}")
        self.move_to_target()

    def move_to_target(self):
        if self.reached_target:
            return  # 已到達目標，不再發送速度

        x, y = self.position
        tx, ty = self.target

        dx = tx - x
        dy = ty - y
        distance = math.hypot(dx, dy)

        if distance < self.distance_tolerance:
            # 到達目標
            self.get_logger().info(f"已到達目標點 ({tx:.3f}, {ty:.3f})")
            self.cmd_pub.publish(Twist())  # 停止
            self.reached_target = True
            return

        # 計算局部座標下的速度（考慮車子 yaw 方向）
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        local_dx = cos_yaw * dx + sin_yaw * dy
        local_dy = -sin_yaw * dx + cos_yaw * dy

        # 限制速度
        if abs(tx) < 0.1:
            speed_x = 0.0
        else:
            speed_x = max(-self.max_linear_speed,
                        min(self.max_linear_speed, local_dx))

        # Y 方向
        if abs(ty) < 0.1:
            speed_y = 0.0
        else:
            speed_y = max(-self.max_linear_speed,
                        min(self.max_linear_speed, local_dy))

        twist = Twist()
        twist.linear.x = speed_x
        twist.linear.y = speed_y
        twist.angular.z = 0.0  # 不旋轉，直接平移

        self.cmd_pub.publish(twist)
        # self.get_logger().info(
        #     f"距離目標: {distance:.3f}, local_dx: {local_dx:.3f}, local_dy: {local_dy:.3f}"
        # )

def main():
    rclpy.init()
    node = GoToPoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('手動停止')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
