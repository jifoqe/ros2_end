#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <nlohmann/json.hpp>
// #include "rclcpp/qos.hpp"
#include <algorithm>

using json = nlohmann::json;
using std::placeholders::_1;

class ESP32Subscriber : public rclcpp::Node
{
public:
    ESP32Subscriber() : Node("micro_ros_receiver")
    {
        // auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        // qos.reliable();

        // ===== Subscriber =====
        esp32_subscription_3 = this->create_subscription<std_msgs::msg::String>(
            "esp32_topic",
            10,
            std::bind(&ESP32Subscriber::topic_esp32_data_3, this, _1)
        );

        // ===== Publisher =====
        move_pub_ = this->create_publisher<std_msgs::msg::String>("/move_data", 10);
        // move_pub_ = this->create_publisher<std_msgs::msg::String>("/move_data",qos);
        web_output_pub_ = this->create_publisher<std_msgs::msg::String>("/web_output", 10);
    }

private:
    // ===== ESP32 callback =====
    void topic_esp32_data_3(const std_msgs::msg::String::SharedPtr msg)
    {
        char command[64];
        int index = 0;
        int number = 0;
        RCLCPP_INFO(this->get_logger(), "ESP32: '%s'", msg->data.c_str());
        


        sscanf(msg->data.c_str(), "%d,%d,%s", &index, &number, command);
        RCLCPP_INFO(this->get_logger(),"index=%d,number=%d, command=%s",index, number, command);

        if(index == 1 && strcmp(command, "base_move_ok") == 0){
            state[index-1] = number;
        }else if(index == 2 && strcmp(command, "base_move_ok") == 0){
            state[index-1] = number;
        }else if(index == 3 && strcmp(command, "base_move_ok") == 0){
            state[index-1] = number;
        }else if(index == 4 && strcmp(command, "base_move_ok") == 0){
            state[index-1] = number;
        }

        min_number = std::min({state[2], state[3]});

        if(number>min_number){
            return;
        }

        if(index == 1 && strcmp(command, "end") == 0){
            state[index-1] = 0;
        }else if(index == 2 && strcmp(command, "end") == 0){
            state[index-1] = 0;
        }else if(index == 3 && strcmp(command, "end") == 0){
            state[index-1] = 0;
        }else if(index == 4 && strcmp(command, "end") == 0){
            state[index-1] = 0;
        }

        if(state[2] == state[3] && state[2] != 0 && state[3] != 0){
            std_msgs::msg::String good;
            good.data = std::to_string(min_number);
            
            RCLCPP_INFO(this->get_logger(),"同步移動");
            for (int i = 0; i < 100; i++){
                move_pub_->publish(good);
                rclcpp::sleep_for(std::chrono::milliseconds(20)); // 間隔20ms
            }
        }else if(strcmp(command, "base_move_ok") == 0){
            std_msgs::msg::String good;
            good.data = std::to_string(min_number);
            
            RCLCPP_INFO(this->get_logger(),"最慢的要跟上");
            for (int i = 0; i < 100; i++){
                move_pub_->publish(good);
                rclcpp::sleep_for(std::chrono::milliseconds(20)); // 間隔20ms
            }
        }
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
    int min_number = 0;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32Subscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}