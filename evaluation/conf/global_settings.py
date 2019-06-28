ENCODING = 'utf-8'

##########################
# 生产,测试库配置
##########################

# 运行环境[PRODUCT,TEST,LOCAL]
RUNTIME_ENVIRONMENT = 'LOCAL'

if RUNTIME_ENVIRONMENT == 'LOCAL':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '101.201.143.74'
    PRODUCE_DB_USER = 'leidengjun'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://'+PRODUCE_DB_USER+':'+PRODUCE_DB_PASSWD+'@'+PRODUCE_DB_ADDR_OUTTER+'/pingjia?charset=utf8'

    # 测试库
    TEST_DB_ADDR = '101.200.229.249'
    TEST_DB_USER = 'pingjia'
    TEST_DB_PASSWD = 'De32wsxC'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://'+TEST_DB_USER+':'+TEST_DB_PASSWD+'@'+TEST_DB_ADDR+'/china_used_car_estimate?charset=utf8'
    TEST_PINGJIA_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/pingjia?charset=utf8'

elif RUNTIME_ENVIRONMENT == 'PRODUCT':
    # 生产库外网
    PRODUCE_DB_ADDR_OUTTER = '10.45.138.200'
    PRODUCE_DB_USER = 'valuate_user'
    PRODUCE_DB_PASSWD = 'ldj_DEV_~!_0705'
    PRODUCE_PINGJIA_ENGINE = 'mysql+pymysql://' + PRODUCE_DB_USER + ':' + PRODUCE_DB_PASSWD + '@' + PRODUCE_DB_ADDR_OUTTER + '/pingjia?charset=utf8'

    # 生产库
    TEST_DB_ADDR = '10.174.11.103:3306'
    TEST_DB_USER = 'valuate_user'
    TEST_DB_PASSWD = 'ldj_DEV_~!_0705'
    TEST_PINGJIA_ENGINE = 'mysql+pymysql://' + TEST_DB_USER + ':' + TEST_DB_PASSWD + '@' + TEST_DB_ADDR + '/china_used_car_estimate?charset=utf8'

###########################
# 模型预测配置
###########################
# 畅销程度系数
PROFITS = {'A': (0.05, 0.095, 0.027, 0.02, 0.12, 0.08, 0.09, 0.006, -0.01),
           'B': (0.05, 0.114, 0.031, 0.025, 0.14, 0.10, 0.10, 0.007, -0.01),
           'C': (0.05, 0.134, 0.02, 0.03, 0.16, 0.12, 0.11, 0.003, -0.01)}

# 各车况因素的系数
CAR_CONDITION = ['excellent', 'good', 'fair', 'bad']
CAR_CONDITION_COEFFICIENT = {'excellent': 1.04, 'good': 1, 'fair': 0.95, 'bad': 0.89}
CAR_CONDITION_COEFFICIENT_VALUES = [1.04, 1, 0.95, 0.89]

# 交易方式
INTENT_TYPE = ['sell', 'buy', 'release', 'private', 'lowest', 'cpo', 'replace', 'auction', 'avg-buy', 'avg-sell']
INTENT_TYPE_CAN = ['buy', 'buy', 'buy', 'buy', 'buy', 'buy', 'buy', 'buy', 'buy', 'buy']

# 返回类型
RETURN_RECORDS = 'records'
RETURN_NORMAL = 'normal'

###########################
# 异常类型
###########################
ERROR_PARAMS = 'PARAMS'
ERROR_SQL = 'SQL'