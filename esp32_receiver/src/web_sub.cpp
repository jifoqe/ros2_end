//網頁通訊 subscriber
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "string"

class WebSubscriber : public rclcpp::Node {
public:
WebSubscriber() : Node("web_subscriber") {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
        "/web_input", 10,
        [this](std_msgs::msg::String::SharedPtr msg) {
            RCLCPP_INFO(this->get_logger(), "收到: '%s'", msg->data.c_str());

            auto out_msg = std_msgs::msg::String();
            out_msg.data = msg->data;
            publisher_->publish(out_msg);
        }
    );
    publisher_ = this->create_publisher<std_msgs::msg::String>("/web_ros_date", 10);
}

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WebSubscriber>());
    rclcpp::shutdown();
    return 0;
}
