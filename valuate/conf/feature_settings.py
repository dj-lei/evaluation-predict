# 排量
VOLUME = {'0': 1, '0-1': 2, '1-2': 3, '2-3': 4, '3-4': 5, '4plus': 6}

# 变速箱
CONTROL = {'手动': 1, '自动': 2}

# 国标
EMISSION_STANDARD = {'国1': 1, '国2': 2, '国3': 3, '国3带OBD': 3, '国4': 4, '国4 ': 4, '国4(京5)': 4, '京5': 4, '国5': 5, '国6': 6,
                     '欧1': 1, '欧2': 2, '欧3': 3, '欧4': 4, '欧5': 5, '欧6': 6}

# 价格段
PRICE = {'0-5': 1, '5-10': 2, '10-20': 3, '20-30': 4, '30-50': 5, '50plus': 6}

# 国系
BRAND_AREA = {'德系': 1, '法系': 2, '日系': 3, '韩系': 4, '美系': 5, '欧系': 6, '国产': 7}

# 合资/国产/进口
ATTRIBUTE = {'进口': 1, '合资': 2, '国产': 3}

# 流行度
POPULARITY = {'A': 1, 'B': 2, 'C': 3}

# 重要的城市
DATA_ORIGIN = {'origin_data': 1, 'predict_common_data': 2}

# 统计模型训练特征
STATISTICAL_MODEL_TRAIN_FEATURE = ['brand_area_encode','attribute_encode','used_years','province_encode','popularity_encode','price_bn_encode','volume_encode','control_encode','emission_standard_encode','discount']
BRAND_MODEL_TRAIN_FEATURE = ['model_detail_slug_encode', 'province_encode', 'used_years']
STATISTICAL_MODEL_TARGET_FEATURE = 'hedge'

# 统计异常
STATISTICAL_EXCEPT_RESIDUALS_SEQUENCE_PROVINCE = 'except_residuals_sequence_province'
STATISTICAL_EXCEPT_USED_YEAR_HEDGE_PROVINCE = 'except_used_year_hedge_province'
STATISTICAL_EXCEPT_UPPER_LOWER_CONFIGURE = 'except_upper_lower_configure'