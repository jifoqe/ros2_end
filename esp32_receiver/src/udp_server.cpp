#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <string>

class UdpServerNode : public rclcpp::Node
{
public:
    UdpServerNode() : Node("udp_server_node")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("/web_output", 10);
        // 每秒定時傳訊息給 web
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            [this]() {
                std_msgs::msg::String msg;
                msg.data = "Hello from C++ node!";
                publisher_->publish(msg);
                RCLCPP_INFO(this->get_logger(), "已傳送給 web: %s", msg.data.c_str());
            }
        );
        // esp32_date.data = "Hello from C++ node!";
        // publisher_->publish(esp32_date);

        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/web_ros_date", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::string s = msg->data;   // string
                RCLCPP_INFO(this->get_logger(), "收到: '%s'", s.c_str());
                web_date = std::stoi(s);
            }
        );

        RCLCPP_INFO(this->get_logger(), "UDP Server Node started.");

        // 建立 socket
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create socket");
            exit(1);
        }

        // 綁定本地 port
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(server_port_);

        if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Bind failed");
            exit(1);
        }

        RCLCPP_INFO(this->get_logger(), "Server listening on port %d", server_port_);

        // 設定 ESP32 IP & port
        client_addr_.sin_family = AF_INET;
        client_addr_.sin_port = htons(client_port_);
        inet_pton(AF_INET, client_ip_.c_str(), &client_addr_.sin_addr);

        // 建立定時器：每 1 秒發送數字
        send_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&UdpServerNode::send_numbers, this)
        );

        // 建立定時器：每 100ms 嘗試接收資料
        recv_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&UdpServerNode::receive_data, this)
        );
    }

    ~UdpServerNode() {
        close(sockfd_);
    }

private:
    void send_numbers()
    {
        if(web_date!="0"){
            std::string msg = web_date;
            sendto(sockfd_, msg.c_str(), msg.size(), 0,
                (struct sockaddr*)&client_addr_, sizeof(client_addr_));
            web_date = "0";
            RCLCPP_INFO(this->get_logger(), "Sent: %s", msg.c_str());
        }

    }

    void receive_data()
    {
        char buffer[1024] = {0};
        sockaddr_in sender_addr{};
        socklen_t addr_len = sizeof(sender_addr);

        int n = recvfrom(sockfd_, buffer, sizeof(buffer)-1, MSG_DONTWAIT,
                         (struct sockaddr*)&sender_addr, &addr_len);

        if (n > 0) {
            buffer[n] = '\0';
            std::string data(buffer);
            RCLCPP_INFO(this->get_logger(), "Received from ESP32: %s", data.c_str());
        }
    }

    int sockfd_;
    int server_port_ = 8888;        // 本機 port
    std::string client_ip_ = "192.168.1.100"; // ESP32 IP，請改成實際 IP
    int client_port_ = 8888;        // ESP32 接收 port
    sockaddr_in client_addr_;

    std::string web_date = "0";
    std_msgs::msg::String esp32_date;

    rclcpp::TimerBase::SharedPtr send_timer_;
    rclcpp::TimerBase::SharedPtr recv_timer_;
    rclcpp::TimerBase::SharedPtr timer_;


    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UdpServerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
