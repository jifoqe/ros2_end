#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/string.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>
#include <cctype>

struct ClientInfo {
    int id; // 車號
    std::string ip; // 例: 192.168.1.100
    int port;       // 例: 8888
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
                msg.data = "131"; // 範例訊息
                publisher_->publish(msg);
                RCLCPP_INFO(this->get_logger(), "已傳送給 Web: %s", msg.data.c_str());
            }
        );

        // 訂閱 Web 訊息
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/web_ros_date", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                web_date_ = msg->data;
                RCLCPP_INFO(this->get_logger(), "收到 Web 指令: '%s'", web_date_.c_str());
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
        clients_ = {
            {1, "10.189.119.201", 8888, {}},
            {2, "10.189.119.202", 8888, {}},
            {3, "10.189.119.203", 8888, {}},
            {4, "10.189.119.204", 8888, {}}
        };

        for(auto &c : clients_){
            c.addr.sin_family = AF_INET;
            c.addr.sin_port = htons(c.port);
            inet_pton(AF_INET, c.ip.c_str(), &c.addr.sin_addr);
        }

        // 建立定時器：每 100ms 嘗試發送訊息給指定 ESP32
        send_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
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
        if(web_date_.empty() || web_date_ == "0"){
            return;
        }

        std::string msg = web_date_;

        if(msg.size() < 2){
            RCLCPP_WARN(this->get_logger(), "Web command格式錯誤: '%s'", msg.c_str());
            web_date_ = "0";
            return;
        }

        char action = msg[0]; // 功能碼
        std::vector<int> car_ids;

        for(size_t i = 1; i < msg.size(); i++){
            if(isdigit(msg[i])){
                car_ids.push_back(msg[i] - '0');
            }
        }

        for(auto id : car_ids){
            bool found = false;
            for(auto &c : clients_){
                if(c.id == id){
                    sendto(sockfd_,
                           &action,
                           1, // 單字元
                           0,
                           (struct sockaddr*)&c.addr,
                           sizeof(c.addr));
                    RCLCPP_INFO(this->get_logger(),
                        "Send action '%c' -> car %d (%s:%d)",
                        action, c.id, c.ip.c_str(), c.port);
                    found = true;
                    break;
                }
            }
            if(!found){
                RCLCPP_WARN(this->get_logger(), "Car id %d 不存在!", id);
            }
        }

        web_date_ = "0"; // 清空
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
            RCLCPP_INFO(this->get_logger(), "Received from %s:%d -> %s", ip, port, data.c_str());
        }
    }

private:
    int sockfd_;
    int server_port_ = 8888;  // 本機 UDP port
    std::vector<ClientInfo> clients_; // 多台 ESP32
    std::string web_date_ = "0";//發送給 ESP32 的資料

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