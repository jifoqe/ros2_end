#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <std_msgs/msg/string.hpp>

class ArucoCameraNode : public rclcpp::Node
{
public:
  ArucoCameraNode()
  : Node("aruco_camera_node")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10,
      std::bind(&ArucoCameraNode::image_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image", 10);
    gray_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image_gray", 10);
    position_publisher_ = this->create_publisher<std_msgs::msg::String>("/web_position_data", 10);

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // 相機內參 (請換成你自己的校正值)
    camera_matrix_ = (cv::Mat1d(3,3) << 600, 0, 320,
                                        0, 600, 240,
                                        0, 0, 1);
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

    marker_length_ = 0.05; // 公尺
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids);

      if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(frame, corners, ids);

        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);

        for (size_t i = 0; i < ids.size(); i++) {
          cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], 0.05);

          // 只取 x, y
          double x = tvecs[i][0];
          double y = tvecs[i][1];

          // 翻轉 y 座標，使左下為原點
          y = -y;

          // 旋轉向量 -> Euler
          cv::Mat R;
          cv::Rodrigues(rvecs[i], R);
          double sy = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
          double yaw;
          if (sy > 1e-6) {
              // roll  = atan2(R.at<double>(2,1), R.at<double>(2,2));
              // pitch = atan2(-R.at<double>(2,0), sy);
              yaw   = atan2(R.at<double>(1,0), R.at<double>(0,0));
          } else {
              // roll  = atan2(-R.at<double>(1,2), R.at<double>(1,1));
              // pitch = atan2(-R.at<double>(2,0), sy);
              yaw   = 0;
          }
          // roll  = roll  * 180.0 / CV_PI;
          // pitch = pitch * 180.0 / CV_PI;
          yaw   = -yaw   * 180.0 / CV_PI;

          RCLCPP_INFO(this->get_logger(),
              "ID:%d -> x: %.3f, y: %.3f, yaw: %.1f",
              ids[i], x, y, yaw);

          std_msgs::msg::String pos_msg;
          // 這裡做成簡單 CSV: "id,x,y,yaw"
          char buf[128];
          std::snprintf(buf, sizeof(buf), "%d,%.6f,%.6f,%.3f", ids[i], x, y, yaw);
          pos_msg.data = std::string(buf);
          position_publisher_->publish(pos_msg);
        }
      }

      cv::imshow("Camera", frame);
      cv::imshow("Gray Image", gray);
      cv::waitKey(1);

      auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
      publisher_->publish(*out_msg);

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
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr position_publisher_;

  cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  double marker_length_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
