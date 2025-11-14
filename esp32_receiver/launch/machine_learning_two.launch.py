from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='esp32_receiver',
            executable='machine_learning_two.py',  # 不要加 .py
            name='multi_agent_sim',
            output='screen'
        )
    ])
