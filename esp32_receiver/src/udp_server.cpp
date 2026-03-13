#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

struct ClientInfo {
    std::string ip;
    int port;
    sockaddr_in addr;
};

class UdpServerNode : public rclcpp::Node
{
public:
    UdpServerNode() : Node("udp_server_node")
    {
        // Publisher 發訊息給 Web
        publisher_ = this->create_publisher<std_msgs::msg::String>("/web_output", 10);

        // 每秒定時傳訊息給 Web
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            [this]() {
                std_msgs::msg::String msg;
                msg.data = "1"; //完成
                publisher_->publish(msg);
                RCLCPP_INFO(this->get_logger(), "已傳送給 web: %s", msg.data.c_str());
            }
        );

        // 訂閱 Web 訊息
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/web_ros_date", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                web_date = msg->data;
                RCLCPP_INFO(this->get_logger(), "收到 Web 指令: '%s'", web_date.c_str());
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

        // 初始化 4 台 ESP32 客戶端資訊
        clients = {
            {"10.46.204.201", 8888, {}},
            {"10.46.204.202", 8888, {}},
            {"10.46.204.203", 8888, {}},
            {"10.46.204.204", 8888, {}}
        };

        for(auto &c : clients){
            c.addr.sin_family = AF_INET;
            c.addr.sin_port = htons(c.port);
            inet_pton(AF_INET, c.ip.c_str(), &c.addr.sin_addr);
        }

        // 建立定時器：每 1 秒發送訊息給所有 ESP32
        send_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&UdpServerNode::send_numbers, this)
        );

        // 建立定時器：每 100ms 嘗試接收 ESP32 發送的資料
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
        if(web_date != "0") {
            std::string msg = web_date;
            for(auto &c : clients){
                sendto(sockfd_, msg.c_str(), msg.size(), 0, (struct sockaddr*)&c.addr, sizeof(c.addr));
                RCLCPP_INFO(this->get_logger(), "Sent to %s:%d -> %s", c.ip.c_str(), c.port, msg.c_str());
            }
            web_date = "0";
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
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &sender_addr.sin_addr, ip, INET_ADDRSTRLEN);
            int port = ntohs(sender_addr.sin_port);
            std::string data(buffer);
            RCLCPP_INFO(this->get_logger(),"Received from %s:%d -> %s", ip, port, data.c_str());
        }
    }

    int sockfd_;
    int server_port_ = 8888;  // 本機 UDP port
    std::vector<ClientInfo> clients; // 多台 ESP32
    std::string web_date = "0";//發送給esp32的資料

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