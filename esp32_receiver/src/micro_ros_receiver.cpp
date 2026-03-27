#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1;

class ESP32Subscriber : public rclcpp::Node
{
public:
    ESP32Subscriber() : Node("micro_ros_receiver")
    {
        // 訂閱 ESP32 Topic
        esp32_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "esp32_topic_1",
            10,
            std::bind(&ESP32Subscriber::topic_esp32_data, this, _1)
        );

        // 訂閱 演算法的資料 Topic
        algo_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "web_ros_date",
            10,
            std::bind(&ESP32Subscriber::algorithm_data, this, _1)
        );

        // 發布 控制指令給 ESP32 的 Topic
        publisher_1_ = this->create_publisher<std_msgs::msg::String>("/move_data_1", 10);
        publisher_2_ = this->create_publisher<std_msgs::msg::String>("/move_data_2", 10);
        publisher_3_ = this->create_publisher<std_msgs::msg::String>("/move_data_3", 10);
        publisher_4_ = this->create_publisher<std_msgs::msg::String>("/move_data_4", 10);
    }

private:
    // 訂閱 ESP32 Topic
    void topic_esp32_data(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "Received from ESP32: '%s'", msg->data.c_str());
    }

    // 訂閱 演算法的資料 Topic
    void algorithm_data(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received from Algorithm: '%s'", msg->data.c_str());

        // 建立要發送的訊息
        std::string data = msg->data;
        size_t len = data.size();
        std_msgs::msg::String out_msg;

        // 根據字元決定發送到哪個 publisher
        if(len > 1 && data[1] == '1') {
            out_msg.data = std::string() + data[0] + data[1];  // 第一組字元
            publisher_1_->publish(out_msg);
            RCLCPP_INFO(this->get_logger(), "[Algorithm] Command '1' -> move_data_1 published: '%s'", out_msg.data.c_str());
        }
        if(len > 3 && data[3] == '2') {
            out_msg.data = std::string() + data[2] + data[3];  // 第二組字元
            publisher_2_->publish(out_msg);
            RCLCPP_INFO(this->get_logger(), "[Algorithm] Command '2' -> move_data_2 published: '%s'", out_msg.data.c_str());
        }
        if(len > 5 && data[5] == '3') {
            out_msg.data = std::string() + data[4] + data[5];  // 第三組字元
            publisher_3_->publish(out_msg);
            RCLCPP_INFO(this->get_logger(), "[Algorithm] Command '3' -> move_data_3 published: '%s'", out_msg.data.c_str());
        }
        if(len > 7 && data[7] == '4') {
            out_msg.data = std::string() + data[6] + data[7];  // 第四組字元
            publisher_4_->publish(out_msg);
            RCLCPP_INFO(this->get_logger(), "[Algorithm] Command '4' -> move_data_4 published: '%s'", out_msg.data.c_str());
        }
    }

    // 訂閱用變數
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr esp32_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr algo_subscription_;

    // Publisher 用變數
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_1_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_2_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_3_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_4_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32Subscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}