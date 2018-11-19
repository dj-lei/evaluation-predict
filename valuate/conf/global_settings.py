ENCODING = 'utf-8'

##########################
# 生产,测试库配置
##########################

# 运行环境[PRODUCT,TEST,LOCAL]
RUNTIME_ENVIRONMENT = 'LOCAL'

if RUNTIME_ENVIRONMENT == 'LOCAL':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '101.201.148.49'
    # PRODUCE_DB_ADDR_OUTTER = '101.201.143.74'
    PRODUCE_DB_USER = 'leidengjun'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/pingjia?charset=utf8'
    PRODUCE_VALUATE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/valuate?charset=utf8'
    PRODUCE_DATASOURCE_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/datasource?charset=utf8'

    # 测试库
    TEST_DB_ADDR = '101.200.229.249'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://'+TEST_DB_USER+':'+TEST_DB_PASSWD+'@'+TEST_DB_ADDR+'/valuate?charset=utf8'

elif RUNTIME_ENVIRONMENT == 'TEST':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '10.45.138.200'
    PRODUCE_DB_USER = 'leidengjun'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/pingjia?charset=utf8'
    PRODUCE_VALUATE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/valuate?charset=utf8'
    PRODUCE_DATASOURCE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/datasource?charset=utf8'

    # 测试库
    TEST_DB_ADDR = '10.44.206.161'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/valuate?charset=utf8'

elif RUNTIME_ENVIRONMENT == 'PRODUCT':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '10.45.138.200'
    PRODUCE_DB_USER = 'valuate_user'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!_0705'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/pingjia?charset=utf8'
    PRODUCE_VALUATE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/valuate?charset=utf8'
    PRODUCE_DATASOURCE_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/datasource?charset=utf8'

    # 生产库
    TEST_DB_ADDR = '10.174.11.103:3306'
    TEST_DB_USER = 'valuate_user'
    TEST_DB_PASSWD = 'ldj_DEV_~!_0705'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/valuate?charset=utf8'

###########################
# 估值系統相關表
###########################
# 预测数据表
VALUATE_PREDICT_DATA = 'valuate_predict_data_alter'
# 预测数据表
VALUATE_RESIDUALS_DATA = 'valuate_residuals_data'
# 异常历史表
VALUATE_ERROR_HISTORY = 'valuate_error_history'
# 车型款型匹配表
VALUATE_MODEL_DETAIL_MAP = 'valuate_model_detail_map'
# c2b推算表
VALUATE_C2B_CALCULATE_DATA = 'valuate_c2b_calculate_data'
# c2b最小二乘法参数表
VALUATE_C2B_LEAST_SQUARES_DATA = 'valuate_c2b_least_squares_data'
# 人工新增记录表
VALUATE_ADD_DATA = 'valuate_add_data'
# 估值操作记录表
VALUATE_OPERATION_HISTORY = 'valuate_operation_history'
# 车型款型匹配表
VALUATE_DIFFERENCE_DATA = 'valuate_difference_data'
VALUATE_DIFFERENCE_DATA_FEATURE = ['car_id','brand_name','model_name','detail_model','model_detail_slug','mile','year','month','city','source_type',
                                   'domain','url','sold_time','price_bn','price','predict_price','artificial_price','process_status','expired_time','update_time','create_time']

###########################
# 模型训练配置
###########################
# 价格预测模型训练需要的特征
TRAIN_FEATURE = ['model_detail_slug_encode', 'used_years', 'province_encode']

# 目标特征
TARGET_FEATURE = 'hedge'

###########################
# 模型预测配置
###########################
# 公里数阈值和范围
# 正常行驶的车辆以一年2.5万公里为正常基数，低于2.5万公里的价格的浮动在+3.5%以内
# 大于2.5万公里的若每年的平均行驶里程大于2.5万公里小于5万公里价格浮动在-3.5-7.5%
# 若年平均形式里程大于5万公里及以上影响价格在-7.5-12.5%之间
MILE_THRESHOLD_2_5 = 2.5
MILE_THRESHOLD_5 = 5
MILE_THRESHOLD_10 = 10

# 畅销程度系数
PROFITS = {'A': (0.06, 0.11, 0.027, 0.02, 0.12, 0.08, 0.09, 0.006, -0.01),
           'B': (0.05, 0.13, 0.031, 0.025, 0.14, 0.10, 0.10, 0.007, -0.01),
           'C': (0.05, 0.15, 0.02, 0.03, 0.16, 0.12, 0.11, 0.003, -0.01)}


###########################
# 异常类型
###########################
ERROR_PARAMS = 'PARAMS'
ERROR_SQL = 'SQL'
ERROR_MANUAL = 'MANUAL '
ERROR_FE = 'FE'
ERROR_TRAIN = 'TRAIN'
ERROR_CRON = 'CRON'
ERROR_PREDICT_MODEL = 'PREDICT_MODEL'
ERROR_NO_DATA = 'NO_DATA'
ERROR_PREDICT_RECORD = 'PREDICT_RECORD'

ERROR_SQL_DESC_QUERY = 'QUERY'
ERROR_SQL_DESC_UPDATE = 'UPDATE'
ERROR_SQL_DESC_INSERT = 'INSERT'
ERROR_SQL_DESC_C2B_DATA_EXCEPT = 'C2B_DATA_EXCEPT'
ERROR_MANUAL_DESC_MOD = 'TABLE_MOD'
ERROR_FE_DESC_ARTIFICIAL = 'ARTIFICIAL_EXCEPT'
ERROR_FE_DESC_REVIEW = 'NO_REVIEW_DATA'
ERROR_FE_NO_MODELS = 'NO_MODELS'

###########################
# 标志位
###########################
FE_EXECUTE = 'execute'
FE_FILTER = 'filter'
RESPONSE_OK = 'OK'
RESPONSE_FAIL = 'FAIL'
PESPONSE_MSG_NEED_POST = 'Need POST!'
PESPONSE_MSG_CALL_MANAGER = 'Call the manager!'


###########################
# 操作类型
###########################
OPERATION_UPDATE = 'update'

###########################
# 人工参与相关特征
###########################
ARTIFICIAL_ALTER_FEATURE = ['car_id', 'model_detail_slug', 'mile', 'year', 'month', 'city', 'source_type', 'domain', 'artificial_price', 'sold_time', 'process_status']
ARTIFICIAL_ADD_FEATURE = ['model_detail_slug', 'mile', 'year', 'month', 'city', 'source_type', 'price', 'sold_time', 'status']
ARTIFICIAL_TRAIN_FEATURE = ['city', 'hedge_rate', 'max_year', 'model_detail_slug', 'price', 'price_bn', 'price_classify', 'source_type', 'used_years']

###########################
# 流行度分类
###########################
POPULARITY_CATEGORY = ['A', 'B', 'C']
