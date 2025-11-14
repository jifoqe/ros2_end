# ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/robot1/cmd_vel

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_path = os.path.join(get_package_share_directory('esp32_receiver'))
    xacro_file = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    
    launch_nodes = []

    # 宣告 use_sim_time
    launch_nodes.append(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true')
    )

    # 設定五台車的位置
    # start_positions = [(-1, 0), (-0.5, 0), (0, 0), (0.5, 0), (1, 0)]
    # start_positions = [(0, 0)]
    start_positions = [(0, 0), (4, 0), (8, 0),
                       (0, -4), (4, -4), (8, -4),
                       (0, -8), (4, -8), (8, -8),
                    ]
    # start_positions = [(random.uniform(-1.1, 1.0), random.uniform(-0.8, 0.6))]

    for i, (x, y) in enumerate(start_positions):
        ns = f"robot{i+1}"

        # ✅ 這裡一定要用 mappings={'ns': ns}
        robot_description_config = xacro.process_file(
            xacro_file,
            mappings={'ns': ns}
        )

        # robot_state_publisher
        params = {
            'robot_description': robot_description_config.toxml(),
            'use_sim_time': use_sim_time
        }

        node_rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='state_publisher',
            output='screen',
            namespace=ns,
            parameters=[params]
        )

        spawn_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', f'/{ns}/robot_description',
                '-entity', ns,
                '-robot_namespace', f'/{ns}',
                '-x', str(x),
                '-y', str(y),
                '-z', '0.1'
            ],
            output='screen'
        )

        launch_nodes.append(node_rsp)
        launch_nodes.append(spawn_node)

    return LaunchDescription(launch_nodes)