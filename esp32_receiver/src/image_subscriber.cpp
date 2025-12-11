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
  : Node("aruco_camera_node"),
    origin_set_(false),
    origin_id_(0) // ID 0 當作世界原點
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", 10,
      std::bind(&ArucoCameraNode::image_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image", 10);
    gray_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image_gray", 10);
    position_publisher_ = this->create_publisher<std_msgs::msg::String>("/web_position_data", 10);

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // 相機內參
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
          int id = ids[i];

          cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], 0.05);

          cv::Mat R_i;
          cv::Rodrigues(rvecs[i], R_i);
          cv::Mat t_i = (cv::Mat1d(3,1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]);

          // --- ID=0 世界原點 ---
          if (id == origin_id_ && !origin_set_) {
            R0_ = R_i.clone();
            t0_ = t_i.clone();
            origin_set_ = true;

            // 世界座標永遠 0,0,0
            double x_cm = 0.0;
            double y_cm = 0.0;
            double yaw_deg = 0.0;

            // 計算 pixel 座標
            double Xc = t_i.at<double>(0,0);
            double Yc = t_i.at<double>(1,0);
            double Zc = t_i.at<double>(2,0);
            double u = camera_matrix_.at<double>(0,0) * Xc / Zc + camera_matrix_.at<double>(0,2);
            double v = camera_matrix_.at<double>(1,1) * Yc / Zc + camera_matrix_.at<double>(1,2);

            RCLCPP_INFO(this->get_logger(), "Origin (ID 0) -> pixel(u,v): %.2f, %.2f", u, v);

            std_msgs::msg::String pos_msg;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%d,%.2f,%.2f,%.1f,%.2f,%.2f", id, x_cm, y_cm, yaw_deg, u, v);
            pos_msg.data = std::string(buf);
            position_publisher_->publish(pos_msg);

            continue; // origin 不需再轉換
          }

          // 如果世界原點還沒偵測到，跳過其他 marker
          if (!origin_set_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
              "Origin not detected, skipping ID %d", id);
            continue;
          }

          // --- 計算世界座標 (cm) ---
          cv::Mat t_world = R0_.t() * (t_i - t0_);
          double x_cm = t_world.at<double>(0,0) * 100.0 * 2;
          double y_cm = t_world.at<double>(1,0) * 100.0 * 2;
          double z_cm = t_world.at<double>(2,0) * 100.0;

          // 投影到像素
          double Xc = t_world.at<double>(0,0);
          double Yc = t_world.at<double>(1,0);
          double Zc = t_world.at<double>(2,0);
          // double u = camera_matrix_.at<double>(0,0) * Xc / Zc + camera_matrix_.at<double>(0,2);
          // double v = camera_matrix_.at<double>(1,1) * Yc / Zc + camera_matrix_.at<double>(1,2);
          // 投影到像素 (始終使用相機座標系)
          double u = camera_matrix_.at<double>(0,0) * t_i.at<double>(0,0) / t_i.at<double>(2,0) + camera_matrix_.at<double>(0,2);
          double v = camera_matrix_.at<double>(1,1) * t_i.at<double>(1,0) / t_i.at<double>(2,0) + camera_matrix_.at<double>(1,2);
          
          // 計算 yaw
          cv::Mat R_world = R0_.t() * R_i;
          double yaw_rad = atan2(R_world.at<double>(1,0), R_world.at<double>(0,0));
          double yaw_deg = -yaw_rad * 180.0 / CV_PI;

          RCLCPP_INFO(this->get_logger(),
            "ID:%d -> x: %.2f cm, y: %.2f cm, yaw: %.1f deg, pixel(u,v): %.2f, %.2f",
            id, x_cm, y_cm, yaw_deg, u, v);

          // 發佈資料
          std_msgs::msg::String pos_msg;
          char buf[128];
          std::snprintf(buf, sizeof(buf), "%d,%.2f,%.2f,%.1f,%.2f,%.2f", id, x_cm, y_cm, yaw_deg, u, v);
          pos_msg.data = std::string(buf);
          position_publisher_->publish(pos_msg);
        }
      }

      // 顯示影像
      cv::imshow("Camera", frame);
      cv::imshow("Gray Image", gray);
      cv::waitKey(1);

      auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
      publisher_->publish(*out_msg);

      auto gray_msg = cv_bridge::CvImage(msg->header, "mono8", gray).toImageMsg();
      gray_publisher_->publish(*gray_msg);

    } catch (cv_bridge::Exception &e) {
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

  bool origin_set_;
  int origin_id_;
  cv::Mat R0_;
  cv::Mat t0_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
