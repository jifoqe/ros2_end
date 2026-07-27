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
            "esp32_topic_4",
            10,
            std::bind(&ESP32Subscriber::topic_esp32_data_3, this, _1)
        );

        // ===== Publisher =====
        move_pub_ = this->create_publisher<std_msgs::msg::String>("/move_data_4", 10);
        web_output_pub_ = this->create_publisher<std_msgs::msg::String>("/web_output", 10);
    }

private:
    // ===== ESP32 callback =====
    void topic_esp32_data_3(const std_msgs::msg::String::SharedPtr msg)
    {
        // int index = 3;
        RCLCPP_INFO(this->get_logger(), "Received from ESP32_4: '%s'", msg->data.c_str());
        // state[index] = std::stoi(msg->data);
        // RCLCPP_INFO(this->get_logger(), "獲得資料: '%d'", state[index]);
        // publish_web_output(msg->data);
        // total_move_check(); //都移動好了一起動
    }

    // ===== publish function =====
    void publish_web_output(const std::string &data)
    {
        std_msgs::msg::String msg;
        msg.data = data;

        RCLCPP_INFO(this->get_logger(), "send to web: '%s'", msg.data.c_str());

        web_output_pub_->publish(msg);
    }

    void total_move_check(){
        if(state[0] == 1 && state[1] == 1 && state[2] == 1 && state[3] == 1){
            RCLCPP_INFO(this->get_logger(), "所有動作完成，開始下一步");
            std_msgs::msg::String msg;
            msg.data = "1";
            move_pub_->publish(msg);
        }
    }

    // ===== Subscriber =====
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_3;

    // ===== Publisher =====
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr move_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr web_output_pub_;
    int state[4] = {0}; //0未完成, 1完成
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32Subscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}