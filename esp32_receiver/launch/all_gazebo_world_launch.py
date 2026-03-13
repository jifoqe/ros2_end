from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    package_description = "esp32_receiver"
    urdf_file = 'robot.urdf'
    robot_desc_path = os.path.join(
        get_package_share_directory(package_description),
        "urdf_data",
        urdf_file
    )

    # 讀 URDF
    with open(robot_desc_path, 'r', encoding='utf-8') as infp:
        robot_desc = infp.read()

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher_node',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True
        }]
    )

    # 啟動新版 Gazebo Harmonic（直接呼叫系統指令 gz sim）
    gz_sim_process = ExecuteProcess(
        cmd=['gz', 'sim', '-v', '4', '-r', 'empty.sdf'],
        output='screen'
    )

    # Spawn robot（延遲 2 秒）
    spawn_entity_node = TimerAction(
        period=2.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', 'robot_description',
                '-name', 'quadruped_robot',
                '-allow_renaming', 'true'
            ],
            output='screen'
        )]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        gz_sim_process,
        spawn_entity_node
    ])