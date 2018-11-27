from valuate import *
from valuate.train.model import *

# # 加载车型款型匹配表
# model_detail_map = pd.read_csv(path + 'predict/map/model_detail_map.csv')
# # 加载省份城市匹配表
# province_city_map = pd.read_csv(path + 'predict/map/province_city_map.csv')
# # 加载省份流行度匹配表
# province_popularity_map = pd.read_csv(path + 'predict/map/province_popularity_map.csv')
# # 加载省份流行度匹配表
# promotion_4s_price = pd.read_csv(path + 'predict/map/promotion_4s_price.csv')
# promotion_details = promotion_4s_price.groupby(['model_detail_slug'])['promotion_price'].median().reset_index()
# promotion_details = {k: v for k, v in itertools.zip_longest(list(promotion_details.model_detail_slug.values), list(promotion_details.promotion_price.values))}
