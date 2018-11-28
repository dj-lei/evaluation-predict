from valuate.train import *


def cal_z_score(df):
    """
    计算z得分
    """
    return abs((df['price'] - df['mean_value']) / (df['std_value']))


def func(a, x):
    """
    拟合函数
    """
    k = a
    return k * x


def dist(a, x, y):
    """
    残差
    """
    return func(a, x) - y


def cal_newton_min_param(x, y, c):
    """
    牛顿k参数
    """
    k = 0
    temp = float("inf")
    for step in range(0, 1000):
        k = -step * 0.001
        div = 0
        for i in range(0, len(x)):
            div = div + (c * math.e ** (k * x[i]) - y[i]) ** 2
        if temp < div:
            return k
        else:
            temp = div
    return k


class FeatureEngineering(object):

    def __init__(self):
        # self.train = pd.read_csv(path + '../tmp/train/car_source.csv')
        # self.province_city_map = pd.read_csv(path + 'predict/map/province_city_map.csv')
        self.open_category = pd.read_csv(path + '../tmp/train/open_category.csv')
        brand = self.open_category.loc[(self.open_category['parent'].isnull()), :]
        brand = brand.loc[:, ['name', 'slug', 'brand_area']].rename(
            columns={'name': 'brand_name', 'slug': 'brand_slug'})

        self.open_category = self.open_category.loc[(self.open_category['parent'].notnull()), :]
        self.open_category = self.open_category.rename(columns={'slug': 'global_slug', 'parent': 'brand_slug', 'name': 'global_name'})
        self.open_category = self.open_category.loc[:, ['brand_slug', 'global_slug', 'global_name']]
        self.open_category = self.open_category.merge(brand, how='left', on=['brand_slug'])
        self.open_model_detail = pd.read_csv(path + '../tmp/train/open_model_detail.csv')
        self.open_model_detail = self.open_model_detail.merge(self.open_category, how='left', on=['global_slug'])
        self.open_model_detail = self.open_model_detail.rename(columns={'year': 'online_year'})
        self.model_global_mean = self.open_model_detail.loc[:,
                            ['brand_name', 'global_name', 'detail_model', 'brand_slug', 'global_slug', 'model_detail_slug', 'online_year', 'price_bn',
                             'volume', 'control', 'emission_standard', 'brand_area']]
        self.open_model_detail = self.open_model_detail.loc[:,
                            ['brand_name', 'global_name', 'detail_model', 'model_detail_slug', 'year',
                             'volume', 'control', 'emission_standard', 'brand_area']]

    ###########################
    # 数据清洗
    ###########################
    def handle_data_quality(self):
        """
        处理数据质量
        """
        self.train = self.train.loc[(self.train['domain'].isin(['guazi.com', 'renrenche.com'])) & (self.train['model_detail_slug'].notnull()), :]
        self.train.reset_index(inplace=True, drop=True)
        self.train = self.train.merge(self.open_model_detail, how='left', on=['model_detail_slug'])
        self.train = self.train.drop(['popularity', 'expired_at', 'sold_time', 'source_type', 'status', 'domain'], axis=1)
        # 删掉price<0的记录
        self.train = self.train[self.train['price'] > 0]
        self.train = self.train[self.train['price_bn'] > 0]
        # 删掉不在20年车龄,月份1-12之外记录
        self.train = self.train[(self.train['year'] <= datetime.datetime.now().year)]
        self.train = self.train[self.train['month'].isin(list(np.arange(1, 13)))]
        # 删除掉未知城市,未知款型的记录
        cities = list(set(self.province_city_map.city.values))
        models = list(set(self.open_model_detail.model_detail_slug.values))
        self.train = self.train[self.train['city'].isin(cities)]
        self.train = self.train[self.train['model_detail_slug'].isin(models)]
        self.train.reset_index(inplace=True, drop='index')

    def handle_data_preprocess(self):
        """
        数据预处理
        1.删除离群点
        """
        # 删除数据量小于12的款型
        detail_num = self.train.groupby(['model_detail_slug'])['detail_model'].count().reset_index()
        detail_num = detail_num.loc[(detail_num['detail_model'] >= 12), :]
        self.train = self.train.loc[(self.train['model_detail_slug'].isin(list(set(detail_num.model_detail_slug.values)))), :]
        self.train.reset_index(inplace=True, drop=True)

        # 根据款型计算均值
        mean_value = self.train.groupby(['model_detail_slug'])['price'].mean().reset_index().rename(
            columns={'price': 'mean_value'})
        self.train = self.train.merge(mean_value, how='left', on=['model_detail_slug'])

        # 根据款型计算标准差
        std_value = self.train.groupby(['model_detail_slug'])['price'].std().reset_index().rename(
            columns={'price': 'std_value'})
        self.train = self.train.merge(std_value, how='left', on=['model_detail_slug'])

        # 计算Z得分，根据阈值删除离群点
        self.train['z_score'] = self.train.apply(cal_z_score, axis=1)
        self.train = self.train.loc[(self.train['z_score'] <= 1.5), :]
        self.train.reset_index(inplace=True, drop=True)
        self.train = self.train.drop(['mean_value', 'std_value', 'z_score'], axis=1)
        self.train.to_csv(path + '../tmp/train/train.csv', index=False)

    def generate_model_global_mean(self):
        """
        生成全款型全国均值模型
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算均值
        median_price = self.train.groupby(['brand_area', 'brand_slug', 'model_slug', 'model_detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        median_price.to_csv(path + '../tmp/train/test.csv', index=False)
        # 根据年限,统计指导价差值的价格差
        model_year = median_price.loc[:, ['model_slug', 'online_year']]
        model_year = model_year.drop_duplicates(['model_slug', 'online_year']).reset_index(drop=True)

        count = 0
        result = pd.DataFrame([], columns=['model_slug', 'online_year', 'price_bn_div', 'price_div'])
        for model_slug, online_year in model_year.loc[:, ['model_slug', 'online_year']].values:
            temp = median_price.loc[(median_price['model_slug'] == model_slug) & (median_price['online_year'] == online_year), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            for i in range(1, len(temp)):
                price_bn_div = temp.loc[i, 'price_bn'] - temp.loc[0, 'price_bn']
                price_div = temp.loc[i, 'median_price'] - temp.loc[0, 'median_price']
                result.loc[count, ['model_slug', 'online_year', 'price_bn_div', 'price_div']] = [model_slug, online_year, price_bn_div, price_div]
                count = count + 1

        # 拟合车系年份线性k参数
        model_slug = self.model_global_mean.loc[:, ['global_slug', 'brand_area']].drop_duplicates(['global_slug', 'brand_area']).reset_index(drop=True).rename(columns={'global_slug': 'model_slug'})
        result = result.merge(model_slug, how='left', on=['model_slug'])
        brand_area_year = result.loc[:, ['brand_area', 'online_year']].drop_duplicates(['brand_area', 'online_year']).reset_index(drop=True)

        count = 0
        k_param = pd.DataFrame([], columns=['brand_area', 'online_year', 'k'])
        for brand_area, online_year in brand_area_year.loc[:, ['brand_area', 'online_year']].values:
            temp = result.loc[(result['brand_area'] == brand_area) & (result['online_year'] == online_year), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            param = [0]
            var = leastsq(dist, param, args=(np.array(list(temp.price_bn_div.values)), np.array(list(temp.price_div.values))))
            k = var[0][0]

            k_param.loc[count, ['brand_area', 'online_year', 'k']] = [brand_area, online_year, k]
            count = count + 1
        k_param = k_param.sort_values(by=['brand_area', 'online_year']).reset_index(drop=True)
        k_param['used_years'] = datetime.datetime.now().year - k_param['online_year']
        k_param.loc[(k_param['used_years'] < 0), 'used_years'] = 0

        # 计算牛顿k参数
        c = median(k_param.loc[(k_param['used_years'] == 0), 'k'].values)
        k = cal_newton_min_param(list(k_param.used_years.values), list(k_param.k.values), c)
        used_years_k_param = [[i, c * math.e ** (k * i)] for i in range(0, 21)]
        print(used_years_k_param)

        # self.model_global_mean = self.model_global_mean.merge(median_price.loc[:, ['model_detail_slug', 'median_price']], how='left', on=['model_detail_slug'])
        # self.model_global_mean = self.model_global_mean.sort_values(by=['brand_name', 'global_name', 'online_year', 'price_bn'])
        # self.model_global_mean.to_csv(path + '../tmp/train/test.csv', index=False)

    def execute(self):
        """
        执行
        """
        # self.handle_data_quality()
        # self.handle_data_preprocess()
        self.generate_model_global_mean()