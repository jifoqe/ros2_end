from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # rosbridge_server 的 launch 檔案位置
    rosbridge_launch = os.path.join(
        get_package_share_directory('rosbridge_server'),
        'launch',
        'rosbridge_websocket_launch.xml'
    )

    return LaunchDescription([
        # 1️⃣ 啟動 rosbridge server (websocket)
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(rosbridge_launch)
        ),

        #開啟鏡頭
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {"video_device": "/dev/video0"},   # ← 在這裡指定你的外接相機
                # {"image_size": [640, 480]},        # 可選
                {"pixel_format": "YUYV"}           # 可選
            ]
        ),

        #讓網頁可以透過src顯示影像
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            output='screen'
        ),

        # 顯示網頁送過來的資料
        Node(
            package='esp32_receiver',
            executable='web_sub',
            name='web_sub',
            output='screen'
        ),

        #讓網頁可以顯示影像
        Node(
            package='esp32_receiver',
            executable='image_subscriber',
            name='image_subscriber',
            output='screen'
        ),
    ])
