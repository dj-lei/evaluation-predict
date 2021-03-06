-- MySQL Script generated by MySQL Workbench
-- 2018年12月24日 星期一 16时36分03秒
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema china_used_car_estimate
--
-- 中国二手车估值系统数据库
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `china_used_car_estimate` DEFAULT CHARACTER SET utf8 ;
USE `china_used_car_estimate` ;

CREATE TABLE IF NOT EXISTS `china_used_car_estimate`.`base_standard_open_category` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `source_id` int(11) DEFAULT NULL COMMENT '来源id',
  `name` varchar(50) DEFAULT NULL COMMENT '品牌/型号名称',
  `alias` varchar(50) DEFAULT NULL COMMENT '别名',
  `slug` varchar(32) DEFAULT NULL COMMENT '品牌/型号slug(唯一标示)',
  `url` varchar(200) DEFAULT NULL COMMENT '品牌url',
  `parent` varchar(32) DEFAULT NULL COMMENT '型号对应品牌slug',
  `checker_runtime_id` int(11) DEFAULT NULL,
  `keywords` varchar(100) DEFAULT NULL COMMENT '关键字',
  `classified` varchar(32) DEFAULT NULL COMMENT '级别',
  `classified_url` varchar(200) DEFAULT NULL,
  `slug_global` varchar(32) DEFAULT NULL,
  `logo_img` varchar(200) DEFAULT NULL COMMENT 'logo',
  `mum` varchar(32) DEFAULT NULL COMMENT '厂商',
  `first_letter` varchar(1) DEFAULT NULL COMMENT '首字母',
  `has_detailmodel` int(11) NOT NULL COMMENT '是否有详细款型',
  `starting_price` decimal(10,1) DEFAULT NULL COMMENT '起步价格(万元)',
  `classified_slug` varchar(128) DEFAULT NULL,
  `thumbnail` varchar(200) DEFAULT NULL COMMENT '缩略图',
  `pinyin` varchar(32) DEFAULT NULL COMMENT '拼音',
  `status` varchar(1) DEFAULT NULL COMMENT 'A:刚添加,Y:确定投入使用,DELETE:标记为需要删除的品牌',
  `attribute` varchar(10) DEFAULT NULL COMMENT '进口/合资/国产',
  `units` int(11) NOT NULL COMMENT '参与统计的该型号车源数量',
  `popular` varchar(1) DEFAULT NULL COMMENT 'A:畅销,B:一般,C:冷门',
  `on_sale` tinyint(1) DEFAULT '0' COMMENT '是否在售',
  `score` int(11) NOT NULL DEFAULT '0' COMMENT '权重',
  `normalized_name` varchar(255) DEFAULT NULL COMMENT '规则化车系，用于显示',
  `brand_area` varchar(20) DEFAULT NULL COMMENT '产地:德系/欧系/美系/日系/国产/法系/韩系',
  PRIMARY KEY (`id`),
  KEY i_slug(slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='公平价品牌型号对应关系';

CREATE TABLE IF NOT EXISTS `china_used_car_estimate`.`base_standard_open_model_detail` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `source_id` int(11) DEFAULT NULL,
  `checker_runtime_id` int(11) DEFAULT NULL COMMENT '个人车源清理状态标志',
  `old_dmodel` varchar(50) DEFAULT NULL COMMENT '旧dmodel名称',
  `detail_model` varchar(50) DEFAULT NULL COMMENT '中文款型',
  `detail_model_slug` varchar(50) DEFAULT NULL COMMENT '款型slug',
  `price_bn` decimal(10,2) DEFAULT NULL COMMENT '款型slug(万元)',
  `cont_vprice` decimal(10,2) DEFAULT NULL,
  `url` varchar(200) DEFAULT '',
  `global_slug` varchar(32) DEFAULT NULL COMMENT '全局slug',
  `domain` varchar(32) DEFAULT NULL COMMENT '域名',
  `status` varchar(1) DEFAULT 'Y' COMMENT 'A:刚添加的款型,Y:确定可投入使用的款型,D:标记为需要删除的款型',
  `year` int(11) NOT NULL COMMENT '年款',
  `has_param` varchar(1) DEFAULT NULL COMMENT 'Y:有配置参数信息,N:无配置参数信息',
  `volume` decimal(5,1) DEFAULT NULL COMMENT '排量',
  `vv` decimal(5,1) DEFAULT NULL,
  `listed_year` int(11) NOT NULL COMMENT '上市年份',
  `delisted_year` int(11) DEFAULT NULL COMMENT '退市年份',
  `control` varchar(32) DEFAULT NULL COMMENT '变速箱',
  `emission_standard` varchar(20) DEFAULT NULL COMMENT '排放标准',
  `volume_extend` varchar(20) DEFAULT NULL COMMENT '排量扩展',
  `simple_model` varchar(50) DEFAULT NULL COMMENT '最简款型',
  `continuity_id` int(11) DEFAULT NULL COMMENT '标记款型连续,同一组款型标记一致',
  `body_model` varchar(10) DEFAULT NULL COMMENT '车身型式',
  `created_on` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY i_detail_model_slug(detail_model_slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='GPJ款型关系';

CREATE TABLE IF NOT EXISTS `china_used_car_estimate`.`base_car_deal_history` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `brand_zh` varchar(32) DEFAULT NULL COMMENT '品牌名称',
  `model_zh` varchar(32) DEFAULT NULL COMMENT '车系名称',
  `model_detail_zh` varchar(50) DEFAULT NULL COMMENT '款型名称',
  `brand_slug` varchar(32) DEFAULT NULL COMMENT '品牌slug',
  `model_slug` varchar(32) DEFAULT NULL COMMENT '车系slug',
  `model_detail_slug` varchar(32) DEFAULT NULL COMMENT '款型slug',
  `year` int(11) NOT NULL DEFAULT '0' COMMENT '年份',
  `month` int(11) NOT NULL DEFAULT '6' COMMENT '月份',
  `mile` decimal(5,2) DEFAULT NULL COMMENT '公里数',
  `volume` decimal(5,1) DEFAULT NULL COMMENT '排量',
  `color` varchar(32) DEFAULT NULL COMMENT '颜色',
  `control` varchar(32) DEFAULT NULL COMMENT '变速器',
  `province` varchar(50) DEFAULT NULL COMMENT '省份',
  `city` varchar(50) DEFAULT NULL COMMENT '城市',
  `deal_time` datetime DEFAULT NULL COMMENT '交易时间',
  `price` decimal(10,2) DEFAULT NULL COMMENT '价格',
  `create_time` datetime NOT NULL COMMENT '创建时间',
  `deal_type` varchar(20) DEFAULT NULL COMMENT '交易类型',
  `source` varchar(50) DEFAULT NULL COMMENT '来源',
  `condition` varchar(20) DEFAULT NULL COMMENT '车况',
  PRIMARY KEY (`id`),
  KEY i_model_detail_slug(model_detail_slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='车源历史交易数据';

CREATE TABLE IF NOT EXISTS `china_used_car_estimate`.`valuate_global_model_mean` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `detail_model_slug` varchar(50) DEFAULT NULL COMMENT '全局款型名称',
  `detail_slug` int(11) DEFAULT NULL COMMENT '汽车之家款型id',
  `brand_name` varchar(50) DEFAULT NULL COMMENT '款型名称',
  `brand_slug` int(11) DEFAULT NULL COMMENT '款型id',
  `model_name` varchar(50) DEFAULT NULL COMMENT '车系名称',
  `model_slug` int(11) DEFAULT NULL COMMENT '车系id',
  `detail_name` varchar(256) DEFAULT NULL COMMENT '款型名称',
  `manufacturer` varchar(200) DEFAULT NULL COMMENT '主机厂',
  `price_bn` decimal(10,2) DEFAULT NULL COMMENT '新车指导价',
  `body` varchar(64) DEFAULT NULL COMMENT '车身',
  `energy` varchar(64) DEFAULT NULL COMMENT '能源',
  `online_year` int(4) NOT NULL COMMENT '上市年份',
  `used_years` int(2) DEFAULT NULL COMMENT '使用年数',
  `median_price` decimal(10,2) DEFAULT NULL COMMENT '全国均价',
  `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY i_detail_slug(detail_slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='估值均价表';

CREATE TABLE IF NOT EXISTS `china_used_car_estimate`.`valuate_province_city` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `city` varchar(50) DEFAULT NULL COMMENT '城市',
  `province` varchar(50) DEFAULT NULL COMMENT '省份',
  `k` decimal(10,4) DEFAULT NULL COMMENT 'k系数',
  `b` decimal(10,4) DEFAULT NULL COMMENT 'b系数',
  `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY i_city(city)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='估值城市差异表';

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
