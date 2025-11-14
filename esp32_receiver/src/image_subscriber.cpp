//可以做影像處理 subscriber
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "visualization_msgs/msg/marker.hpp"  // Marker 套件

class ImageSubscriber : public rclcpp::Node
{
public:
  ImageSubscriber()
  : Node("image_subscriber")
  {
    // 訂閱相機 topic
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));

    // 發佈處理後影像
    pub_image_ = this->create_publisher<sensor_msgs::msg::Image>("/processed_image", 10);

    //建地圖
    pub_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("/table_marker", 10);

    // 發佈處理後影像
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("/processed_image", 10);
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    
    try {
      // 轉換 ROS Image → OpenCV Mat
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;


      // 在這裡進行影像處理（這裡以灰階轉換為例）
      cv::Mat gray, blurImg, edge;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(gray, blurImg, cv::Size(5,5), 0);
       // 2. 邊緣偵測
      cv::Canny(blurImg, edge, 50, 150);

        // 3. 找輪廓
      std::vector<std::vector<cv::Point>> contours;
      findContours(edge, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      double maxArea = 0;
      cv::RotatedRect maxRect;

      // 4. 遍歷所有輪廓
      for(size_t i=0; i<contours.size(); i++) {
          // 用最小外接矩形逼近
          cv::RotatedRect rect = cv::minAreaRect(contours[i]);
          double area = rect.size.width * rect.size.height;
          if(area > maxArea) {
              maxArea = area;
              maxRect = rect;
          }

          // 畫出所有矩形（藍色）
          cv::Point2f pts[4];
          rect.points(pts);
          for(int j=0; j<4; j++)
              line(frame, pts[j], pts[(j+1)%4], cv::Scalar(255,0,0), 1);
      }

      if(maxArea > 0){ 
        // 桌子中心點
        // float cx = maxRect.center.x;
        // float cy = maxRect.center.y;

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "camera_frame";  // 根據你的相機 frame
        marker.header.stamp = this->now();
        marker.ns = "table";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Marker 位置
        marker.pose.position.x = maxRect.center.x / 100.0;
        marker.pose.position.y = maxRect.center.y / 100.0;
        marker.pose.position.z = 0.02;  // 桌面厚度

        // 桌子旋轉（Z 軸）
        double yaw = maxRect.angle * CV_PI / 180.0;  // 弧度
        geometry_msgs::msg::Quaternion q;
        q.x = 0.0;
        q.y = 0.0;
        q.z = sin(yaw/2.0);
        q.w = cos(yaw/2.0);

        marker.pose.orientation = q;

        // 桌子尺寸固定 60x50 cm
        marker.scale.x = 0.6;
        marker.scale.y = 0.5;
        marker.scale.z = 0.04;

        // 顏色
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.8f;

        pub_marker_->publish(marker);
      }

      // 5. 標出最大的矩形（綠色）
      cv::Point2f pts[4];
      maxRect.points(pts);
      for(int j=0; j<4; j++)
          line(frame, pts[j], pts[(j+1)%4], cv::Scalar(0,255,0), 3);

      // 發佈影像給 RViz
      auto msg_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
      msg_out->header.stamp = this->now();
      pub_image_->publish(*msg_out);


      // 顯示影像
      cv::imshow("Gray Image", gray);
      cv::imshow("All Rectangles (Blue) / Largest (Green)", frame);
      cv::waitKey(1);

      // 將灰階影像轉回 ROS Image message
      auto gray_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", gray).toImageMsg();
      gray_msg->header.stamp = this->now();

      // 發佈處理後影像
      pub_->publish(*gray_msg);
    }
    catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageSubscriber>());
  rclcpp::shutdown();
  return 0;
}
