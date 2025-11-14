from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    robot_names = ['robot1', 'robot2', 'robot3',
                   'robot4', 'robot5', 'robot6',]
                #    'robot7', 'robot8', 'robot9']  # 想要生成的車子名稱
        # robot_names = ['robot1', 'robot2', 'robot3',
        #            'robot4', 'robot5', 'robot6',
        #            'robot7', 'robot8', 'robot9']  # 想要生成的車子名稱
    nodes = []

    for i, name in enumerate(robot_names):
        nodes.append(
            Node(
                package='esp32_receiver',
                executable='machine_learning.py',
                name=f'multi_agent_sim_{i}',  # node 名稱不能重複
                output='screen',
                arguments=[name]
            )
        )

    return LaunchDescription(nodes)
