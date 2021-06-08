-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 08, 2021 at 11:05 AM
-- Server version: 10.4.14-MariaDB
-- PHP Version: 7.4.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_face`
--

-- --------------------------------------------------------

--
-- Table structure for table `check_temp`
--

CREATE TABLE `check_temp` (
  `p_id` varchar(10) NOT NULL,
  `temp` varchar(10) NOT NULL,
  `check_in` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `check_temp`
--

INSERT INTO `check_temp` (`p_id`, `temp`, `check_in`) VALUES
('John', '36', '1/1/2564'),
('PIN', '36', '2021-06-08 09:11:52.007359'),
('PIN', '36', '2021-06-08 09:11:52.848943'),
('PIN', '36', '2021-06-08 09:11:53.263757'),
('PIN', '36', '2021-06-08 09:11:53.708062'),
('PIN', '36', '2021-06-08 09:11:54.142449'),
('PIN', '36', '2021-06-08 09:11:54.597116'),
('PIN', '36', '2021-06-08 09:11:55.057881'),
('PIN', '36', '2021-06-08 09:11:55.520644'),
('PIN', '36', '2021-06-08 09:11:55.961990'),
('PIN', '36', '2021-06-08 09:11:56.394830'),
('PIN', '36', '2021-06-08 09:15:41.835484'),
('KHEMMIKA', '36', '2021-06-08 09:16:15.061737'),
('PIN', '36', '2021-06-08 09:16:15.963940'),
('KHEMMIKA', '36', '2021-06-08 09:16:16.423139'),
('SANYA', '36', '2021-06-08 09:16:18.807659'),
('PIN', '36', '2021-06-08 09:16:19.368365'),
('KHEMMIKA', '36', '2021-06-08 09:16:19.826659'),
('PIN', '36', '2021-06-08 09:16:20.687534'),
('SANYA', '36', '2021-06-08 09:16:21.216805'),
('KHEMMIKA', '36', '2021-06-08 09:16:21.717676'),
('PIN', '36', '2021-06-08 09:16:25.664595'),
('PIN', '36', '2021-06-08 09:22:18.766671'),
('PIN', '36', '2021-06-08 09:41:45.482239');

-- --------------------------------------------------------

--
-- Table structure for table `login`
--

CREATE TABLE `login` (
  `id` int(5) NOT NULL,
  `user_name` varchar(30) NOT NULL,
  `password` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- Table structure for table `person`
--

CREATE TABLE `person` (
  `p_id` varchar(10) NOT NULL,
  `first_name` varchar(20) NOT NULL,
  `last_name` varchar(20) NOT NULL,
  `age` varchar(2) NOT NULL,
  `job` varchar(20) NOT NULL,
  `img` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `login`
--
ALTER TABLE `login`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `login`
--
ALTER TABLE `login`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
