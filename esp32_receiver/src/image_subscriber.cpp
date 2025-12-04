//可以做影像處理 subscriber
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

class SimpleCameraNode : public rclcpp::Node
{
public:
  SimpleCameraNode()
  : Node("simple_camera_node")
  {
    // 訂閱相機 topic
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&SimpleCameraNode::image_callback, this, std::placeholders::_1));

    // 發佈相機影像
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image", 10);
    gray_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image_fix", 10);
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      // ROS Image → OpenCV Mat
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;

      // 顯示影像 (僅用於 debug)
      cv::imshow("Camera", frame);
      // cv::waitKey(1);

      // 再發佈原影像
      auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
      publisher_->publish(*out_msg);

      // ---- 影像處理：轉灰階 ----
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::imshow("Gray Image", gray);
      // cv::waitKey(1);

      auto gray_msg = cv_bridge::CvImage(msg->header, "mono8", gray).toImageMsg();
      gray_publisher_->publish(*gray_msg);
    }
    catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gray_publisher_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleCameraNode>());
  rclcpp::shutdown();
  return 0;
}

