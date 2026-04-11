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
    origin_id_(0)
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ArucoCameraNode::image_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image", 10);
    gray_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera_image_gray", 10);
    position_publisher_ = this->create_publisher<std_msgs::msg::String>("/web_position_data", 10);

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    camera_matrix_ = (cv::Mat1d(3,3) <<
      658.187275, 0.0, 332.285798,
      0.0, 658.049522, 234.767034,
      0.0, 0.0, 1.0);

    dist_coeffs_ = (cv::Mat1d(1,5) <<
      0.014150,
      -0.111555,
      0.005415,
      -0.000626,
      0.0);

    marker_length_ = 0.096; // meters
  }

private:
  //旋轉矩陣
  cv::Mat T_inv(const cv::Mat &R, const cv::Mat &t)
  {
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
    R_inv.copyTo(T(cv::Rect(0,0,3,3)));
    t_inv.copyTo(T(cv::Rect(3,0,1,3)));
    return T;
  }

  //從 rvec 和 tvec 轉換成 4x4 的變換矩陣
  cv::Mat T_from_rvec_tvec(const cv::Vec3d &rvec, const cv::Vec3d &tvec)
  {
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat t = (cv::Mat1d(3,1) << tvec[0], tvec[1], tvec[2]);

    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(T(cv::Rect(0,0,3,3)));
    t.copyTo(T(cv::Rect(3,0,1,3)));
    return T;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids);//找到aruco marker

      if (ids.empty()) return;

      cv::aruco::drawDetectedMarkers(frame, corners, ids);//畫到frame上匡

      //算出3d位置
      std::vector<cv::Vec3d> rvecs, tvecs;
      cv::aruco::estimatePoseSingleMarkers(
        corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);

      std::map<int, cv::Mat> T_cam_marker;

      //畫出xyz軸
      for (size_t i = 0; i < ids.size(); i++) {
        cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_,rvecs[i], tvecs[i], 0.05);
        T_cam_marker[ids[i]] = T_from_rvec_tvec(rvecs[i], tvecs[i]);
      }

      // -----------------------------
      // STEP 1: define world = marker0
      // -----------------------------
      if (!origin_set_) {
        if (T_cam_marker.count(origin_id_) == 0) {
          return;
        }
        T_world_cam_ = T_inv(T_cam_marker[origin_id_](cv::Rect(0,0,3,3)),T_cam_marker[origin_id_](cv::Rect(3,0,1,3)));
        origin_set_ = true;
      }

      // -----------------------------
      // STEP 2: compute all markers in world frame
      // -----------------------------
      // for (auto &p : T_cam_marker)
      // {
      //   int id = p.first;
      //   cv::Mat T_world_marker = T_world_cam_ * p.second;

      //   cv::Mat R = T_world_marker(cv::Rect(0,0,3,3));
      //   cv::Mat t = T_world_marker(cv::Rect(3,0,1,3));

      //   double x_cm = t.at<double>(0,0) * 100.0;
      //   double y_cm = t.at<double>(1,0) * 100.0;

      //   double yaw = atan2(R.at<double>(1,0), R.at<double>(0,0));
      //   double yaw_deg = yaw * 180.0 / CV_PI;

      //   std_msgs::msg::String msg_out;
      //   char buf[128];
      //   std::snprintf(buf, sizeof(buf), "%d,%.3f,%.3f,%.2f",id, x_cm, y_cm, yaw_deg);
      //   if(id == 2){
      //     RCLCPP_INFO(this->get_logger(), "Car %d: x=%.2f cm, y=%.2f cm, yaw=%.2f deg", id, x_cm, y_cm, yaw_deg);
      //   }
      //   msg_out.data = std::string(buf);
      //   position_publisher_->publish(msg_out);
      // }
      
      std::map<int, double> x_prev_, y_prev_, yaw_prev_;
      bool first_frame_ = true;
      double alpha_ = 0.2;   // 0.1~0.3（越小越穩）
      // ===============================
      // STEP 2: compute all markers in world frame（完整版）
      // ===============================
      for (auto &p : T_cam_marker)
      {
          int id = p.first;

          // world = world_cam * cam_marker
          cv::Mat T_world_marker = T_world_cam_ * p.second;

          cv::Mat R = T_world_marker(cv::Rect(0,0,3,3));
          cv::Mat t = T_world_marker(cv::Rect(3,0,1,3));

          // ===============================
          // 1️⃣ 取出原始數據（cm）
          // ===============================
          double x_now = t.at<double>(0,0) * 100.0;
          double y_now = t.at<double>(1,0) * 100.0;

          double yaw = atan2(R.at<double>(1,0), R.at<double>(0,0));
          double yaw_now = yaw * 180.0 / CV_PI;

          // ===============================
          // 2️⃣ 初始化
          // ===============================
          if (x_prev_.find(id) == x_prev_.end())
          {
              x_prev_[id] = x_now;
              y_prev_[id] = y_now;
              yaw_prev_[id] = yaw_now;
          }

          // ===============================
          // 3️⃣ 防暴衝（跳太大直接忽略）
          // ===============================
          if (std::abs(x_now - x_prev_[id]) > 10.0) x_now = x_prev_[id];
          if (std::abs(y_now - y_prev_[id]) > 10.0) y_now = y_prev_[id];
          if (std::abs(yaw_now - yaw_prev_[id]) > 30.0) yaw_now = yaw_prev_[id];

          // ===============================
          // 4️⃣ 低通濾波（核心）
          // ===============================
          double x_cm = alpha_ * x_now + (1 - alpha_) * x_prev_[id];
          double y_cm = alpha_ * y_now + (1 - alpha_) * y_prev_[id];
          double yaw_deg = alpha_ * yaw_now + (1 - alpha_) * yaw_prev_[id];

          // 更新歷史值
          x_prev_[id] = x_cm;
          y_prev_[id] = y_cm;
          yaw_prev_[id] = yaw_deg;

          // ===============================
          // 5️⃣ LOG（只印車子 id=2）
          // ===============================
          if (id == 2)
          {
              RCLCPP_INFO(this->get_logger(),
                          "Car %d: x=%.2f cm, y=%.2f cm, yaw=%.2f deg",
                          id, x_cm, y_cm, yaw_deg);
          }

          // ===============================
          // 6️⃣ 發布 ROS2 訊息
          // ===============================
          std_msgs::msg::String msg_out;
          char buf[128];
          std::snprintf(buf, sizeof(buf), "%d,%.2f,%.2f,%.2f",
                        id, x_cm, y_cm, yaw_deg);

          msg_out.data = std::string(buf);
          position_publisher_->publish(msg_out);
      }

      cv::imshow("Camera", frame);
      cv::imshow("Gray", gray);
      cv::waitKey(1);

      publisher_->publish(*cv_bridge::CvImage(msg->header,"bgr8",frame).toImageMsg());
      gray_publisher_->publish(*cv_bridge::CvImage(msg->header,"mono8",gray).toImageMsg());

    }
    catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    }
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gray_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr position_publisher_;

  cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  double marker_length_;

  int origin_id_;
  bool origin_set_;
  cv::Mat T_world_cam_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}