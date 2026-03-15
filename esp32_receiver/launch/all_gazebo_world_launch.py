from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # --- URDF 檔案 ---
    package_description = "esp32_receiver"
    urdf_file = 'robot.urdf'
    robot_desc_path = os.path.join(
        get_package_share_directory(package_description),
        "urdf_data",
        urdf_file
    )

    with open(robot_desc_path, 'r', encoding='utf-8') as infp:
        robot_desc = infp.read()

    # --- Robot State Publisher ---
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

    # --- 啟動 Gazebo via ros_gz_sim 官方 launch ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": "-r empty.sdf"  # 運行 empty world
        }.items()
    )

    # --- Spawn robot（延遲 5 秒，保證 Gazebo fully ready） ---
    spawn_entity_node = TimerAction(
        period=5.0,
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

    # --- Optional: RViz node to visualize TF ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            get_package_share_directory(package_description),
            'rviz',
            'robot_tf.rviz'  # 你可以自定義 rviz config
        )],
    )

    return LaunchDescription([
        robot_state_publisher_node,
        gazebo_launch,
        spawn_entity_node,
        # rviz_node  # 若想開 RViz 可以取消註解
    ])