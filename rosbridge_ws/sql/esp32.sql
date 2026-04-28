-- phpMyAdmin SQL Dump
-- version 5.1.1deb5ubuntu1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Apr 28, 2026 at 05:13 PM
-- Server version: 8.0.45-0ubuntu0.22.04.1
-- PHP Version: 8.1.2-1ubuntu2.23

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `esp32`
--

-- --------------------------------------------------------

--
-- Table structure for table `car_state`
--

CREATE TABLE `car_state` (
  `car_number` int NOT NULL COMMENT '車牌編號',
  `line_speed` float NOT NULL COMMENT '直線速度',
  `angle_speed` float NOT NULL COMMENT '角度速度',
  `local_time` datetime NOT NULL COMMENT '更新時間'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='car_base_state(車子基本狀態)';

--
-- Dumping data for table `car_state`
--

INSERT INTO `car_state` (`car_number`, `line_speed`, `angle_speed`, `local_time`) VALUES
(1, 0.5, 0, '2026-04-01 12:42:48'),
(2, 0.5, 0.7, '2025-12-01 14:51:18'),
(3, 0.5, 0, '2025-12-01 22:52:37'),
(4, 2, 1, '2025-12-01 22:06:16'),
(5, 6, 6, '2025-12-01 22:38:30');

-- --------------------------------------------------------

--
-- Table structure for table `draw_square_data`
--

CREATE TABLE `draw_square_data` (
  `id` int NOT NULL,
  `shape` varchar(50) COLLATE utf8mb4_general_ci DEFAULT 'square',
  `date` json DEFAULT NULL,
  `description` text COLLATE utf8mb4_general_ci,
  `serial_number` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='畫正方形資料';

--
-- Dumping data for table `draw_square_data`
--

INSERT INTO `draw_square_data` (`id`, `shape`, `date`, `description`, `serial_number`) VALUES
(1, 'square', '{\"top\": 75, \"left\": 350, \"right\": 575, \"bottom\": 275}', '225x200', 9003),
(2, 'square', '{\"top\": 75, \"left\": 350, \"right\": 575, \"bottom\": 275}', '225x200', 4723),
(3, 'square', '{\"top\": 225, \"left\": 225, \"right\": 400, \"bottom\": 350}', '175x125', 5157),
(4, 'square', '{\"top\": 150, \"left\": 425, \"right\": 450, \"bottom\": 175}', '25x25', 9971),
(5, 'square', '{\"top\": 175, \"left\": 325, \"right\": 475, \"bottom\": 325}', '150x150', 9561),
(6, 'square', '{\"top\": 175, \"left\": 225, \"right\": 450, \"bottom\": 375}', '225x200', 4184),
(7, 'square', '{\"top\": 175, \"left\": 225, \"right\": 425, \"bottom\": 350}', '200x175', 2829),
(8, 'square', '{\"top\": 200, \"left\": 325, \"right\": 450, \"bottom\": 300}', '125x100', 2486),
(9, 'square', '{\"top\": 325, \"left\": 150, \"right\": 475, \"bottom\": 450}', '325x125', 2199),
(10, 'square', '{\"top\": 225, \"left\": 350, \"right\": 525, \"bottom\": 350}', '175x125', 9412),
(11, 'square', '{\"top\": 175, \"left\": 350, \"right\": 625, \"bottom\": 400}', '300x200', 4467),
(12, 'square', '{\"top\": 200, \"left\": 375, \"right\": 550, \"bottom\": 400}', '175x200', 935),
(13, 'square', NULL, '75x50', 9995),
(14, 'square', NULL, '275x175', 2284),
(15, 'square', NULL, '400x375', 6882),
(16, 'square', NULL, '150x200', 153),
(17, 'square', NULL, '125x75', 2464),
(18, 'square', NULL, '175x200', 911),
(19, 'square', NULL, '125x175', 7047),
(20, 'square', NULL, '125x150', 4844),
(21, 'square', NULL, '100x75', 3861),
(22, 'square', NULL, '75x50', 6017),
(23, 'square', NULL, '200x100', 5234),
(24, 'square', '\"{\\\"left\\\":250,\\\"right\\\":475,\\\"top\\\":225,\\\"bottom\\\":325}\"', '225x100', 9922),
(25, 'square', '\"{\\\"left\\\":75,\\\"right\\\":675,\\\"top\\\":75,\\\"bottom\\\":500}\"', '600x425', 8250),
(26, 'square', '\"{\\\"left\\\":275,\\\"right\\\":725,\\\"top\\\":225,\\\"bottom\\\":550}\"', '450x325', 5974),
(27, 'square', '{\"top\": 275, \"left\": 125, \"right\": 300, \"bottom\": 450}', '175x175', 7076),
(28, 'square', '{\"top\": 25, \"left\": 125, \"right\": 350, \"bottom\": 275}', '225x250', 9587),
(29, 'square', '{\"top\": 225, \"left\": 125, \"right\": 350, \"bottom\": 350}', '225x125', 224),
(30, 'square', '{\"top\": 75, \"left\": 125, \"right\": 425, \"bottom\": 350}', '300x275', 6354),
(31, 'square', '{\"top\": 275, \"left\": 100, \"right\": 400, \"bottom\": 525}', '300x250', 6066),
(32, 'square', '{\"top\": 150, \"left\": 175, \"right\": 400, \"bottom\": 325}', '225x175', 6606),
(33, 'square', '{\"top\": 250, \"left\": 125, \"right\": 425, \"bottom\": 450}', '300x200', 2012),
(34, 'square', '{\"top\": 325, \"left\": 275, \"right\": 500, \"bottom\": 550}', '225x225', 8235),
(35, 'square', '{\"top\": 300, \"left\": 175, \"right\": 550, \"bottom\": 550}', '375x250', 4351),
(36, 'square', '{\"top\": 225, \"left\": 275, \"right\": 600, \"bottom\": 375}', '325x150', 4021),
(37, 'square', '{\"top\": 50, \"left\": 150, \"right\": 475, \"bottom\": 300}', '325x250', 9280),
(38, 'square', '{\"top\": 150, \"left\": 275, \"right\": 625, \"bottom\": 375}', '350x225', 990),
(39, 'square', '{\"top\": 225, \"left\": 225, \"right\": 450, \"bottom\": 450}', '225x225', 4189),
(40, 'square', '{\"top\": 275, \"left\": 200, \"right\": 400, \"bottom\": 450}', '200x175', 8944),
(41, 'square', '{\"top\": 225, \"left\": 225, \"right\": 425, \"bottom\": 425}', '200x200', 8786),
(42, 'square', '{\"top\": 100, \"left\": 200, \"right\": 475, \"bottom\": 375}', '275x275', 5286),
(43, 'square', '{\"top\": 350, \"left\": 225, \"right\": 375, \"bottom\": 475}', '150x125', 5836),
(44, 'square', '{\"top\": 225, \"left\": 300, \"right\": 550, \"bottom\": 425}', '250x200', 4431),
(45, 'square', '{\"top\": 350, \"left\": 275, \"right\": 525, \"bottom\": 550}', '250x200', 7067),
(46, 'square', '{\"top\": 125, \"left\": 225, \"right\": 525, \"bottom\": 400}', '300x275', 9059),
(47, 'square', '{\"top\": 125, \"left\": 375, \"right\": 575, \"bottom\": 350}', '200x225', 6909),
(48, 'square', '{\"top\": 225, \"left\": 275, \"right\": 525, \"bottom\": 425}', '250x200', 88),
(49, 'square', '{\"top\": 225, \"left\": 150, \"right\": 325, \"bottom\": 375}', '175x150', 1188),
(50, 'square', '{\"top\": 175, \"left\": 250, \"right\": 525, \"bottom\": 375}', '275x200', 2170),
(51, 'square', '{\"top\": 350, \"left\": 200, \"right\": 650, \"bottom\": 550}', '450x200', 640),
(52, 'square', '{\"top\": 350, \"left\": 225, \"right\": 450, \"bottom\": 475}', '225x125', 2811),
(53, 'square', '{\"top\": 225, \"left\": 225, \"right\": 550, \"bottom\": 400}', '325x175', 5292),
(54, 'square', '{\"top\": 200, \"left\": 250, \"right\": 525, \"bottom\": 475}', '275x275', 3447),
(55, 'square', '{\"top\": 300, \"left\": 275, \"right\": 475, \"bottom\": 500}', '200x200', 4799),
(56, 'square', '{\"top\": 175, \"left\": 250, \"right\": 625, \"bottom\": 500}', '375x325', 5766),
(57, 'square', '{\"top\": 375, \"left\": 150, \"right\": 400, \"bottom\": 550}', '250x175', 3550),
(58, 'square', '{\"top\": 400, \"left\": 100, \"right\": 350, \"bottom\": 550}', '250x150', 5576),
(59, 'square', '{\"top\": 350, \"left\": 250, \"right\": 425, \"bottom\": 525}', '175x175', 1385),
(60, 'square', '{\"top\": 300, \"left\": 225, \"right\": 475, \"bottom\": 550}', '250x250', 5520),
(61, 'square', '{\"top\": 225, \"left\": 200, \"right\": 550, \"bottom\": 550}', '350x325', 9964),
(62, 'square', '{\"top\": 175, \"left\": 200, \"right\": 375, \"bottom\": 350}', '175x175', 8047),
(63, 'square', '{\"top\": 400, \"left\": 275, \"right\": 450, \"bottom\": 550}', '175x150', 954),
(64, 'square', '{\"top\": 425, \"left\": 225, \"right\": 425, \"bottom\": 550}', '200x125', 7621),
(65, 'square', '{\"top\": 300, \"left\": 125, \"right\": 375, \"bottom\": 475}', '250x175', 8187),
(66, 'square', '{\"top\": 175, \"left\": 350, \"right\": 550, \"bottom\": 425}', '200x250', 4107),
(67, 'square', '{\"top\": 200, \"left\": 375, \"right\": 550, \"bottom\": 425}', '175x225', 210),
(68, 'square', '{\"top\": 275, \"left\": 200, \"right\": 475, \"bottom\": 475}', '275x200', 4612),
(69, 'square', '{\"top\": 200, \"left\": 200, \"right\": 500, \"bottom\": 425}', '300x225', 2922),
(70, 'square', '{\"top\": 0, \"left\": 75, \"right\": 450, \"bottom\": 375}', '375x375', 8016),
(71, 'square', '{\"top\": 200, \"left\": 300, \"right\": 725, \"bottom\": 525}', '425x325', 601),
(72, 'square', '{\"top\": 100, \"left\": 175, \"right\": 725, \"bottom\": 550}', '550x450', 9125),
(73, 'square', '{\"top\": 200, \"left\": 100, \"right\": 525, \"bottom\": 500}', '425x300', 2112),
(76, 'square', '{\"top\": 0, \"left\": 25, \"right\": 700, \"bottom\": 525}', '675x525', 6505),
(77, 'square', '{\"top\": 150, \"left\": 300, \"right\": 675, \"bottom\": 500}', '375x350', 5231),
(78, 'square', '{\"top\": 225, \"left\": 200, \"right\": 725, \"bottom\": 525}', '525x300', 44),
(79, 'square', '{\"top\": 225, \"left\": 200, \"right\": 725, \"bottom\": 525}', '375x375', 3819);

-- --------------------------------------------------------

--
-- Table structure for table `draw_triangle_data`
--

CREATE TABLE `draw_triangle_data` (
  `id` int NOT NULL,
  `shape` varchar(50) COLLATE utf8mb4_general_ci DEFAULT 'triangle',
  `date` json DEFAULT NULL,
  `description` text COLLATE utf8mb4_general_ci,
  `serial_number` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='畫三角形資料';

--
-- Dumping data for table `draw_triangle_data`
--

INSERT INTO `draw_triangle_data` (`id`, `shape`, `date`, `description`, `serial_number`) VALUES
(1, 'test', '{\"top\": 0, \"left\": 0, \"right\": 100, \"bottom\": 100}', '100x100', 9999);

-- --------------------------------------------------------

--
-- Table structure for table `user_base`
--

CREATE TABLE `user_base` (
  `username` varchar(60) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '使用者',
  `password` char(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '密碼',
  `permissions` enum('1','2','3','4','5','6','7','8','9') NOT NULL COMMENT '權限'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='使用者基本資料使用者';

--
-- Dumping data for table `user_base`
--

INSERT INTO `user_base` (`username`, `password`, `permissions`) VALUES
('u', '0', '9');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `car_state`
--
ALTER TABLE `car_state`
  ADD PRIMARY KEY (`car_number`);

--
-- Indexes for table `draw_square_data`
--
ALTER TABLE `draw_square_data`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `serial_number` (`serial_number`);

--
-- Indexes for table `draw_triangle_data`
--
ALTER TABLE `draw_triangle_data`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `serial_number` (`serial_number`);

--
-- Indexes for table `user_base`
--
ALTER TABLE `user_base`
  ADD PRIMARY KEY (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `draw_square_data`
--
ALTER TABLE `draw_square_data`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=80;

--
-- AUTO_INCREMENT for table `draw_triangle_data`
--
ALTER TABLE `draw_triangle_data`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
