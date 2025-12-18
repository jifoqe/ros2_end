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
                // RCLCPP_INFO(this->get_logger(), "已回傳: %s", processed.c_str());
            }
        );
    }

private:
    std::string data_process(const std::string &web_data)
    {
        //2 1 0110 0230 0340 0250 0340 0250
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

            int local_x = decode_value(web_data[2], (web_data[3] - '0') * 100 + (web_data[4] - '0') * 10 + (web_data[5] - '0'));
            int local_y = decode_value(web_data[6], (web_data[7] - '0') * 100 + (web_data[8] - '0') * 10 + (web_data[9] - '0'));
            int local_a = decode_value(web_data[10], (web_data[11] - '0') * 100 + (web_data[12] - '0') * 10 + (web_data[13] - '0'));
            int goal_x  = decode_value(web_data[14], (web_data[15] - '0') * 100 + (web_data[16] - '0') * 10 + (web_data[17] - '0'));
            int goal_y  = decode_value(web_data[18], (web_data[19] - '0') * 100 + (web_data[20] - '0') * 10 + (web_data[21] - '0'));
            int goal_a  = decode_value(web_data[22], (web_data[23] - '0') * 100 + (web_data[24] - '0') * 10 + (web_data[25] - '0'));

            RCLCPP_INFO(
                this->get_logger(),
                "解析結果: 功能=%d car=%d local(%d,%d,%d) goal(%d,%d,%d)",
                func - '0', car, local_x, local_y, local_a, goal_x, goal_y, goal_a
            );
            std::string esp32_date = "";
            esp32_date.push_back(web_data[0]);  // 功能
            esp32_date.push_back(web_data[1]);  // 車號
            if(local_a != goal_a){
                if(local_a > goal_a){
                    esp32_date.push_back('6');
                }else{
                    esp32_date.push_back('7');
                }
            }else{
                esp32_date.push_back(move(local_x, local_y, goal_x, goal_y));
            }
            
            // 回傳格式可依你需求調整
            return esp32_date;
        }

        if (func == '3') {
            int car     = web_data[1] - '0';
            int local_x = (web_data[3] - '0') * 100 + (web_data[4] - '0') * 10 + (web_data[5] - '0');
            if(web_data[2] - '0' != 0){
                local_x *= -1;
            }
            int local_y = (web_data[7] - '0') * 100 + (web_data[8] - '0') * 10 + (web_data[9] - '0');
            if(web_data[6] - '0' != 0){
                local_y *= -1;
            }
            int goal_x = (web_data[11] - '0') * 100 + (web_data[12] - '0') * 10 + (web_data[13] - '0');
            if(web_data[10] - '0' != 0){
                goal_x *= -1;
            }
            int goal_y = (web_data[15] - '0') * 100 + (web_data[16] - '0') * 10 + (web_data[17] - '0');
            if(web_data[14] - '0' != 0){
                goal_y *= -1;
            }

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

        if (func == '4') {
            std::array<int, 5> car{};
            std::array<int, 5> local_x{};
            std::array<int, 5> local_y{};
            std::array<int, 5> goal_x{};
            std::array<int, 5> goal_y{};

            for (int i = 0; i < 5; i++) {
                int car_pair = i * 17 + 1;  // 每台車的起始位置

                car[i]      = web_data[car_pair] - '0';
                local_x[i]  = total_car_number(web_data, car_pair + 1); // local x 從 car_pair+1 開始
                local_y[i]  = total_car_number(web_data, car_pair + 5); // local y 從 car_pair+5 開始
                goal_x[i]   = total_car_number(web_data, car_pair + 9); // goal x
                goal_y[i]   = total_car_number(web_data, car_pair + 13); // goal y
            }

            for (int i = 0; i < 5; i++) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Car%d: local(%d,%d) goal(%d,%d)",
                    car[i], local_x[i], local_y[i], goal_x[i], goal_y[i]
                );
            }
            // std::string esp32_date = "";
            std::string esp32_date= "";
            esp32_date.push_back(web_data[0]);  // 功能
            esp32_date.push_back(web_data[1]);  // 車號
            for (int i = 0; i < 5; i++) {
                esp32_date += move(local_x[i], local_y[i], goal_x[i], goal_y[i]);
            }

            // 回傳格式可依你需求調整
            return esp32_date;
        }

        if (func == '5') {
            std::string esp32_date = "";
            esp32_date.push_back(web_data[0]);  // 功能
            esp32_date.push_back(web_data[1]);  // 車號
            esp32_date.push_back('5');       // 動作碼

            // 回傳格式可依你需求調整
            return esp32_date;
        }

        return "UNKNOWN_FUNC";
    }

    //功能4
    int total_car_number(const std::string &web_data, int start) {
        int number = (web_data[start + 1] - '0') * 100 +
                    (web_data[start + 2] - '0') * 10 +
                    (web_data[start + 3] - '0');

        if (web_data[start] - '0' != 0) {
            number *= -1;  // 符號位
        }

        return number;
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
