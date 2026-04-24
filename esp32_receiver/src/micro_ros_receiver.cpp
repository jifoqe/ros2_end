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
        // ===== Subscriber =====
        esp32_subscription_3 = this->create_subscription<std_msgs::msg::String>(
            "esp32_topic_3",
            10,
            std::bind(&ESP32Subscriber::topic_esp32_data_3, this, _1)
        );

        // ===== Publisher =====
        move_pub_ = this->create_publisher<std_msgs::msg::String>("/move_data_1", 10);
        web_output_pub_ = this->create_publisher<std_msgs::msg::String>("/web_output", 10);
    }

private:
    // ===== ESP32 callback =====
    void topic_esp32_data_3(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received from ESP32_3: '%s'",
            msg->data.c_str()
        );
        publish_web_output(msg->data);
    }

    // ===== publish function =====
    void publish_web_output(const std::string &data)
    {
        std_msgs::msg::String msg;
        msg.data = data;

        RCLCPP_INFO(this->get_logger(),
            "send to web: '%s'",
            msg.data.c_str()
        );

        web_output_pub_->publish(msg);
    }

    // ===== Subscriber =====
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_3;

    // ===== Publisher =====
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr move_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr web_output_pub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32Subscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}