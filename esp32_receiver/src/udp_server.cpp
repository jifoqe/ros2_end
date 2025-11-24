#include <rclcpp/rclcpp.hpp>
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
        std::string msg = std::to_string(current_number_);
        sendto(sockfd_, msg.c_str(), msg.size(), 0,
               (struct sockaddr*)&client_addr_, sizeof(client_addr_));

        RCLCPP_INFO(this->get_logger(), "Sent: %s", msg.c_str());

        current_number_++;
        if (current_number_ > 20)
            current_number_ = 11;
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
    int current_number_ = 1;

    rclcpp::TimerBase::SharedPtr send_timer_;
    rclcpp::TimerBase::SharedPtr recv_timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UdpServerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
