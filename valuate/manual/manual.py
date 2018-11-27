from valuate.manual import *


class Manual(object):
    """
    组合模型训练
    """
    def __init__(self):
        self.model_detail_map = []
        self.province_city_map = []
        self.open_province_popularity = []
        self.current_city_encode = []
        self.deal_records = []
        self.domain_category = []

    def generate_model_detail_map(self):
        """
        生成款型详情表,促销表
        """
        cur_time = datetime.datetime.now()
        cur_time = cur_time.strftime('%Y-%m-%d')
        # 生成详情表
        open_category = pd.read_csv(path + '../tmp/train/open_category.csv')
        brand = open_category[open_category['parent'].isnull()]
        brand = brand.loc[:, ['id', 'slug', 'name', 'brand_area']]
        brand = brand.rename(columns={'id': 'brand_slug_id', 'slug': 'parent', 'name': 'brand_name'})
        brand.reset_index(inplace=True)
        model_slug = open_category[open_category['parent'].notnull()]
        model_slug = model_slug.drop(['brand_area'], axis=1)
        model_slug.reset_index(inplace=True)
        model_slug = model_slug.rename(columns={'id': 'model_slug_id', 'slug': 'model_slug', 'name': 'model_name'})
        model_slug = model_slug.merge(brand, how='left', on='parent')
        model_slug = model_slug.loc[:, ['brand_slug_id', 'model_slug_id', 'parent', 'brand_name', 'model_name', 'model_slug', 'attribute', 'brand_area', 'classified', 'classified_slug']]
        model_slug = model_slug.rename(columns={'parent': 'brand_slug'})

        open_model_detail = pd.read_csv(path + '../tmp/train/open_model_detail.csv')
        open_model_detail = open_model_detail.loc[:, ['id', 'global_slug', 'detail_model', 'model_detail_slug', 'price_bn', 'year', 'volume','control','emission_standard', 'status']]
        open_model_detail = open_model_detail.rename(columns={'id': 'model_detail_slug_id', 'global_slug': 'model_slug'})
        open_model_detail = open_model_detail.merge(model_slug, how='left', on='model_slug')
        open_model_detail = open_model_detail[open_model_detail['model_slug_id'].notnull()]
        open_model_detail['brand_slug_id'] = open_model_detail['brand_slug_id'].astype(int)
        open_model_detail['model_slug_id'] = open_model_detail['model_slug_id'].astype(int)
        open_model_detail['create_time'] = cur_time
        open_model_detail = open_model_detail.loc[(open_model_detail['price_bn'] > 0), :]
        open_model_detail['emission_standard'] = open_model_detail['emission_standard'].fillna('国5')
        open_model_detail['control'] = open_model_detail['control'].fillna('自动')
        os.makedirs(os.path.dirname(path + 'predict/map/model_detail_map.csv'), exist_ok=True)
        open_model_detail.to_csv(path + 'predict/map/model_detail_map.csv', index=False)

    def generate_others_predict_relate_tables(self):
        """
        生成预测需要相关表
        """
        cur_time = datetime.datetime.now()
        cur_time = cur_time.strftime('%Y-%m-%d')
        # 生成城市省份匹配表
        open_city = pd.read_csv(path+'../tmp/train/open_city.csv')
        province = open_city[open_city['parent'] == 0]
        province = province.drop('parent', axis=1)
        province = province.rename(columns={'id': 'parent', 'name': 'province'})
        city = open_city[open_city['parent'] != 0]
        city = city.rename(columns={'id': 'city_id', 'name': 'city'})
        self.province_city_map = city.merge(province, how='left', on='parent')
        self.province_city_map = self.province_city_map.rename(columns={'parent': 'province_id'})
        self.province_city_map = self.province_city_map.loc[:, ['province_id', 'province', 'city', 'city_id']]
        self.province_city_map['create_time'] = cur_time
        self.province_city_map.to_csv(path + 'predict/map/province_city_map.csv', index=False)

    def execute(self):
        """
        执行人工处理后续流程
        """
        try:
            self.generate_model_detail_map()
            self.generate_others_predict_relate_tables()
        except Exception:
            raise SqlOperateError(gl.ERROR_MANUAL_DESC_MOD, traceback.print_exc())

