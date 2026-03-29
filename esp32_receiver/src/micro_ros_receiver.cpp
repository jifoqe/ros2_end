#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using std::placeholders::_1;

class ESP32Subscriber : public rclcpp::Node
{
public:
    ESP32Subscriber() : Node("micro_ros_receiver")
    {
        // 訂閱 ESP32 Topic
        // esp32_subscription_1 = this->create_subscription<std_msgs::msg::String>(
        //     "esp32_topic_1",
        //     10,
        //     std::bind(&ESP32Subscriber::topic_esp32_data_1, this, _1)
        // );
        // web_subscription_ = this->create_subscription<std_msgs::msg::String>(
        //     "/web_input", 10,
        //     [this](std_msgs::msg::String::SharedPtr msg) {

        //         RCLCPP_INFO(this->get_logger(), "收到: '%s'", msg->data.c_str());
        //         try {
        //             json j = json::parse(msg->data);

        //             int mode = j["mode"];
        //             int speed = j["speed"];

        //             // direction 是 array
        //             std::string direction = j["direction"][0];

        //             RCLCPP_INFO(this->get_logger(), "mode: %d", mode);
        //             RCLCPP_INFO(this->get_logger(), "direction: %s", direction.c_str());
        //             RCLCPP_INFO(this->get_logger(), "speed: %d", speed);

        //             // cars 是字串陣列
        //             for (auto& car : j["cars"]) {
        //                 std::string car_id = car;
        //                 RCLCPP_INFO(this->get_logger(), "car: %s", car_id.c_str());
        //             }

        //         } catch (json::parse_error& e) {
        //             RCLCPP_ERROR(this->get_logger(), "JSON解析失敗: %s", e.what());
        //         }
        //     }
        // );

        // 發布 控制指令給 ESP32 的 Topic
        // publisher_1_ = this->create_publisher<std_msgs::msg::String>("/move_data_1", 10);
    }

private:
    // 訂閱 ESP32 Topic
    // void topic_esp32_data_1(const std_msgs::msg::String::SharedPtr msg) const
    // {
    //     RCLCPP_INFO(this->get_logger(), "Received from ESP32_1: '%s'", msg->data.c_str());
    // }

    // 訂閱用變數
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_1;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_2;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_3;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_4;
    // rclcpp::Subscription<std_msgs::msg::String>::SharedPtr web_subscription_;

    // Publisher 用變數
    // rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_1_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32Subscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}