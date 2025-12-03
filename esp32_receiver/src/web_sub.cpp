//網頁通訊 subscriber
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "string"

class WebSubscriber : public rclcpp::Node {
public:
    WebSubscriber() : Node("web_subscriber") {

        publisher_ = this->create_publisher<std_msgs::msg::String>("/web_ros_date", 10);

        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/web_input", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {

                RCLCPP_INFO(this->get_logger(), "收到: '%s'", msg->data.c_str());

                std::string processed = data_process(msg->data);

                std_msgs::msg::String out_msg;
                out_msg.data = processed;

                publisher_->publish(out_msg);
                RCLCPP_INFO(this->get_logger(), "已回傳: %s", processed.c_str());
            }
        );
    }

private:
    std::string data_process(const std::string &web_data)
    {
        //2 1 011 023 034 025
        // 資料太短直接回傳錯誤
        if (web_data.size() <= 10) {
            return web_data;
        }

        // 功能
        char func = web_data[0];
        if (func == '1') {
            return "FUNC1_NO_ACTION";
        }

        if (func == '2') {
            int car     = web_data[1] - '0';

            int local_x = decode_value(web_data[2], (web_data[3] - '0') * 10 + (web_data[4] - '0'));
            int local_y = decode_value(web_data[5], (web_data[6] - '0') * 10 + (web_data[7] - '0'));
            int goal_x  = decode_value(web_data[8], (web_data[9] - '0') * 10 + (web_data[10] - '0'));
            int goal_y  = decode_value(web_data[11], (web_data[12] - '0') * 10 + (web_data[13] - '0'));

            RCLCPP_INFO(
                this->get_logger(),
                "解析結果: 功能=%d car=%d local(%d,%d) goal(%d,%d)",
                func - '0', car, local_x, local_y, goal_x, goal_y
            );
            std::string esp32_date = "";
            esp32_date.push_back(web_data[0]);  // 功能
            esp32_date.push_back(web_data[1]);  // 車號
            esp32_date.push_back(move(local_x, local_y, goal_x, goal_y));       // 動作碼

            // 回傳格式可依你需求調整
            return esp32_date;
        }

        return "UNKNOWN_FUNC";
    }

    //功能2
    char move(int local_x, int local_y, int goal_x, int goal_y){
        if(local_x < goal_x){
            return '3';
        }else if(local_x > goal_x){
            return '4';
        }else if(local_y < goal_y){
            return '1';
        }else if(local_y > goal_y){
            return '2';
        }
        return '5';
    }

    //座標格式整理
    int decode_value(char sign, char value)
    {
        int v = value;      // 第二位轉數字

        if (sign == '0')
            return v;             // 0 表示正數

        else if (sign == '1')
            return -v;            // 1 表示負數

        return 0;   // 理論上不會到
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WebSubscriber>());
    rclcpp::shutdown();
    return 0;
}
