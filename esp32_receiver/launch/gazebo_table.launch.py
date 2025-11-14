# import os
# from launch import LaunchDescription
# from launch_ros.actions import Node
# from ament_index_python.packages import get_package_share_directory
# import xacro

# def generate_launch_description():
#     pkg_path = get_package_share_directory('esp32_receiver')
#     xacro_file = os.path.join(pkg_path, 'description', 'gazebo_table.urdf.xacro')

#     # 解析 Xacro
#     robot_description = xacro.process_file(xacro_file).toxml()

#     return LaunchDescription([
#         Node(
#             package='robot_state_publisher',
#             executable='robot_state_publisher',
#             output='screen',
#             parameters=[{'robot_description': robot_description}],
#         ),
#         Node(
#             package='gazebo_ros',
#             executable='spawn_entity.py',
#             arguments=['-topic', 'robot_description', '-entity', 'table_world'],
#             output='screen',
#         ),
#     ])



import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    pkg_path = os.path.join(get_package_share_directory('esp32_receiver'))
    xacro_file = os.path.join(pkg_path, 'description', 'gazebo_table.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    # robot_state_publisher node
    node_rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        # namespace='table_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config.toxml()}]
    )

    # spawn_entity node
    spawn_table = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'table'],
        output='screen'
    )

    return LaunchDescription([
        node_rsp,
        spawn_table
    ])
