-- phpMyAdmin SQL Dump
-- version 5.1.1deb5ubuntu1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Dec 10, 2025 at 03:31 PM
-- Server version: 8.0.44-0ubuntu0.22.04.1
-- PHP Version: 8.1.2-1ubuntu2.22

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

USE `esp32`;

--
-- Database: `zeabur` (formerly esp32)
--

-- --------------------------------------------------------

--
-- Table structure for table `car_state`
--

CREATE TABLE `car_state` (
  `car_number` int NOT NULL COMMENT '車牌編號',
  `line_speed` float NOT NULL COMMENT '直線速度',
  `angle_speed` float NOT NULL COMMENT '角度速度',
  `local_time` datetime NOT NULL COMMENT '更新時間',
  PRIMARY KEY (`car_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='car_base_state(車子基本狀態)';

--
-- Dumping data for table `car_state`
--

INSERT INTO `car_state` (`car_number`, `line_speed`, `angle_speed`, `local_time`) VALUES
(1, 0.5, 0.7, '2025-12-02 09:57:33'),
(2, 0.5, 0.7, '2025-12-01 14:51:18'),
(3, 0.5, 0, '2025-12-01 22:52:37'),
(4, 2, 1, '2025-12-01 22:06:16'),
(5, 6, 6, '2025-12-01 22:38:30');

-- --------------------------------------------------------

--
-- Table structure for table `user_base`
--
CREATE TABLE `user_base` (
  `username` varchar(60) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '使用者',
  `password` char(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '密碼',
  `permissions` enum('1','2','3','4','5','6','7','8','9') NOT NULL COMMENT '權限',
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='使用者基本資料使用者';

--
-- Dumping data for table `user_base`
--

INSERT INTO `user_base` (`username`, `password`, `permissions`) VALUES
('u', '0', '9');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
