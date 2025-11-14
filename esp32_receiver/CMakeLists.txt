cmake_minimum_required(VERSION 3.8)
project(esp32_receiver)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

#OpenCV
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(visualization_msgs REQUIRED)

add_executable(udp_receiver src/udp_receiver.cpp)
ament_target_dependencies(udp_receiver rclcpp std_msgs)
add_executable(subscriber src/subscriber.cpp)
ament_target_dependencies(subscriber rclcpp std_msgs)

# web_sub
add_executable(web_sub src/web_sub.cpp)
ament_target_dependencies(web_sub rclcpp std_msgs)

#OpenCV
add_executable(image_subscriber src/image_subscriber.cpp)
ament_target_dependencies(image_subscriber rclcpp sensor_msgs cv_bridge OpenCV)

ament_target_dependencies(image_subscriber
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  visualization_msgs
)

install(
  PROGRAMS
    launch/machine_learning.py   # ← 路徑要正確，假設你的檔案放在 esp32_receiver/scripts 下
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS
    launch/machine_learning_two.py   # ← 路徑要正確，假設你的檔案放在 esp32_receiver/scripts 下
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS
    launch/gazebo_image.py   # ← 確認你的腳本路徑正確
  DESTINATION lib/${PROJECT_NAME}
)
install(
  PROGRAMS
    launch/open_pkl.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  image_subscriber
  web_sub 
  subscriber
  udp_receiver
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY description launch woilds config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
