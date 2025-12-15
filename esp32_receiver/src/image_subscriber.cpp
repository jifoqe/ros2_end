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
    origin_id_(0) // 把 ID 0 當作 world origin，若要改其他 ID，改這裡
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

        // 每個偵測到的 marker
        for (size_t i = 0; i < ids.size(); i++) {
          cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], 0.05);

          // 取得 marker -> camera 的旋轉矩陣 R_i 與平移向量 t_i
          cv::Mat R_i;
          cv::Rodrigues(rvecs[i], R_i); // R_i : rotation matrix from marker_i to camera
          cv::Mat t_i = (cv::Mat1d(3,1) << tvecs[i][0], tvecs[i][1], tvecs[i][2]); // in meters

          int id = ids[i];

          // 如果是 world origin marker，儲存它的 R0, t0（marker -> camera）
          if (id == origin_id_) {
              R0_ = R_i.clone();
              t0_ = t_i.clone();
              origin_set_ = true;

              // 將 origin 自己在 world 中定義為 (0,0,0) 並發布（可選）
              double x0_cm = 0.0;
              double y0_cm = 0.0;
              double yaw0 = 0.0;

              // RCLCPP_INFO(this->get_logger(), "Origin (ID %d) set.", origin_id_);

              std_msgs::msg::String pos_msg;
              char buf0[128];
              std::snprintf(buf0, sizeof(buf0), "%d,%.6f,%.6f,%.3f", id, x0_cm, y0_cm, yaw0);
              pos_msg.data = std::string(buf0);
              position_publisher_->publish(pos_msg);

              continue; // origin 不需要再轉換
          }

          // 如果 world origin 還沒被偵測到，跳過其他 marker（或你也可以選擇發 camera frame pose）
          if (!origin_set_) {
              // optional: 發 camera frame 下的原始資料（若需要）
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Origin (ID %d) not detected yet, skipping ID %d", origin_id_, id);
              continue;
          }

          // 計算 world (marker0) 座標下的位移：
          // T_cam_marker = [R_i, t_i]
          // world frame 為 marker0 的 frame
          // T_world_marker_i = inv(T_cam_marker0) * T_cam_marker_i
          // 因為 inv(T_cam_marker0) = [R0_.t(), -R0_.t() * t0_]
          cv::Mat R0_inv = R0_.t();
          cv::Mat t_world = R0_inv * (t_i - t0_); // translation of marker_i in world frame (meters)
          cv::Mat R_world = R0_inv * R_i;         // rotation of marker_i in world frame

          // 轉成你慣用的單位 / 方向（你原本使用 cm 且把 y 翻轉成上為正）
          double x_cm = t_world.at<double>(0,0) * 100.0 * 2;
          double y_cm = t_world.at<double>(1,0) * 100.0 * 2;
          double z_cm = t_world.at<double>(2,0) * 100.0;

          // 如果你想要 y 向上為正（左下為原點樣式），把 y 取反（這次是對所有 marker 一致）
          // y_cm = -y_cm;

          // 從 R_world 計算 yaw（以 Z 軸為旋轉軸），跟你原本的方法類似
          double yaw_rad = atan2(R_world.at<double>(1,0), R_world.at<double>(0,0));
          double yaw_deg = -yaw_rad * 180.0 / CV_PI; // 保持之前顯示習慣的符號（如需要可調整）

          // RCLCPP_INFO(this->get_logger(),
          //     "ID:%d -> x: %.3f cm, y: %.3f cm, yaw: %.1f deg (z: %.3f cm)",
          //     id, x_cm, y_cm, yaw_deg, z_cm);

          std_msgs::msg::String pos_msg;
          // CSV: "id,x_cm,y_cm,yaw_deg"
          char buf[128];
          std::snprintf(buf, sizeof(buf), "%d,%.6f,%.6f,%.3f", id, x_cm, y_cm, yaw_deg);
          pos_msg.data = std::string(buf);
          position_publisher_->publish(pos_msg);
        } // for each marker
      } // if ids not empty

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

  // world origin 相關
  bool origin_set_;
  int origin_id_;
  cv::Mat R0_; // rotation matrix marker0 -> camera
  cv::Mat t0_; // translation marker0 -> camera
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}