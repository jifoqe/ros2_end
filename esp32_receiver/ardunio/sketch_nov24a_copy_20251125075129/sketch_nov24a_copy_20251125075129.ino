#include <WiFi.h>
#include <WiFiUdp.h>

// const char* ssid = "realme GT NEO 3";
// const char* pass = "00000000";

const char* ssid = "Al007L";
const char* pass = "ai007ai007";

WiFiUDP udp;
const char* host = "192.168.50.116"; // ROS2 server IP
const int port = 8888;               // ROS2 server port

// ⭐ ESP32 固定 IP
IPAddress local_IP(192, 168, 50, 202);
IPAddress gateway(192, 168, 50, 1);
IPAddress subnet(255, 255, 255, 0);

HardwareSerial STM32Serial(2); 

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  // ⭐ esp32 IP
  WiFi.config(local_IP, gateway, subnet);
  WiFi.begin(ssid, pass);
  STM32Serial.begin(115200, SERIAL_8N1, 16, 17);

  int maxTry = 20;
  while (WiFi.status() != WL_CONNECTED && maxTry > 0) {
    delay(1000);
    Serial.print(".");
    maxTry--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("Local IP is: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connect failed!");
  }

  udp.begin(port); // 開啟 UDP 監聽本地 port（用於接收 server 回傳資料）
}

void loop() {
  // 1️⃣ 傳送 1~10
  for (int i = 11; i <= 20; i++) {
    udp.beginPacket(host, port);
    udp.print(i);
    udp.endPacket();
    Serial.print("Sent: ");
    Serial.println(i);

    // 2️⃣ 嘗試接收回傳資料
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[128];
      int len = udp.read(buffer, sizeof(buffer) - 1);
      if (len > 0) {
        buffer[len] = 0; // 結尾
        Serial.print("Received from server: ");
        Serial.println(buffer);
        STM32Serial.println(buffer);//傳給stm32
      }
    }

    delay(1000);
  }
}











































#include <WiFi.h>

#include <micro_ros_platformio.h>

#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <std_msgs/msg/string.h>

// ===== WiFi 設定 =====
char ssid[] = "realme GT NEO 3";
char password[] = "00000000";

// ===== Agent（你電腦IP）=====
IPAddress agent_ip(10, 189, 119, 112);  // ⭐ 改成你的電腦IP
const uint16_t agent_port = 8888;

// ===== micro-ROS =====
rcl_publisher_t publisher;
rcl_node_t node;
rclc_support_t support;
rcl_allocator_t allocator;

std_msgs__msg__String msg;

void setup() {
  Serial.begin(115200);
  delay(2000);

  // ===== 1️⃣ 先手動連 WiFi（穩定）=====
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // ===== 2️⃣ 設定 micro-ROS WiFi =====
  set_microros_wifi_transports(
    ssid,
    password,
    agent_ip,
    agent_port
  );

  delay(2000);

  // ===== 3️⃣ 初始化 micro-ROS =====
  allocator = rcl_get_default_allocator();

  rclc_support_init(&support, 0, NULL, &allocator);

  rclc_node_init_default(&node, "esp32_node", "", &support);

  rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String),
    "esp32_topic"
  );

  // ===== 4️⃣ 初始化 message =====
  msg.data.data = (char*)malloc(50);
  msg.data.capacity = 50;

  Serial.println("micro-ROS ready!");
}

void loop() {
  static int count = 0;

  sprintf(msg.data.data, "Hello %d from ESP32", count++);
  msg.data.size = strlen(msg.data.data);

  rcl_publish(&publisher, &msg, NULL);

  Serial.println(msg.data.data);

  delay(1000);
}
