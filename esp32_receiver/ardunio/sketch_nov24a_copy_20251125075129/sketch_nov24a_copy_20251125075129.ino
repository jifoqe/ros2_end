#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "realme GT NEO 3";
const char* pass = "00000000";

WiFiUDP udp;
const char* host = "10.225.173.122"; // ROS2 server IP
const int port = 8888;               // ROS2 server port

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);

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
  for (int i = 1; i <= 10; i++) {
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
      }
    }

    delay(1000);
  }
}
