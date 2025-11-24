//發送訊號 node topic
#include <rclcpp/rclcpp.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <string>

class UdpReceiverNode : public rclcpp::Node
{
public:
    UdpReceiverNode() : Node("udp_receiver_node")
    {
        RCLCPP_INFO(this->get_logger(), "UDP Receiver Node started.");

        // 建立 socket
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create socket");
            exit(1);
        }

        // 綁定 IP 和 port
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY; // 接收任何來源
        addr.sin_port = htons(port_);

        if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Bind failed");
            exit(1);
        }

        // 建立定時器，每 100ms 嘗試接收資料
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&UdpReceiverNode::receive_data, this)
        );
    }

    ~UdpReceiverNode() {
        close(sockfd_);
    }

private:
    void receive_data()
    {
        char buffer[1024] = {0};
        sockaddr_in sender_addr{};
        socklen_t addr_len = sizeof(sender_addr);

        int n = recvfrom(sockfd_, buffer, sizeof(buffer)-1, MSG_DONTWAIT,
                         (struct sockaddr*)&sender_addr, &addr_len);

        if (n > 0) {
            buffer[n] = '\0'; // 字串結尾
            std::string data(buffer);
            RCLCPP_INFO(this->get_logger(), "Received: %s", data.c_str());
        }
        // 如果 n == -1，表示暫時沒有資料，不用處理
    }

    int sockfd_;
    int port_ = 8888; // 與 ESP32 的 port 一致
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UdpReceiverNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
