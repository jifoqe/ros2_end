#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')
        self.bridge = CvBridge()
        # 訂閱相機 topic
        self.subscription = self.create_subscription(
            Image,
            '/sky_cam/sky_cam_sensor/image_raw',  # 你的相機 topic
            self.image_callback,
            10
        )
        self.get_logger().info('Camera Processor Node Started')

    def image_callback(self, msg):
        try:
            # 轉換 ROS Image → OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 顯示影像
            cv2.imshow('Sky Camera', cv_image)
            cv2.waitKey(1)
            # 這裡可以加你的影像處理程式
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
