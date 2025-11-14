//網頁通訊 subscriber
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class WebSubscriber : public rclcpp::Node {
public:
    WebSubscriber() : Node("web_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/web_input", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(this->get_logger(), "收到: '%s'", msg->data.c_str());
            }
        );
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WebSubscriber>());
    rclcpp::shutdown();
    return 0;
}
