from valuate.train import *


def cal_z_score(df):
    """
    计算z得分
    """
    return abs((df['price'] - df['mean_value']) / (df['std_value']))


def cal_use_mile_per_months(df):
    """
    计算每月公里数
    """
    return df['mile'] / ((datetime.datetime.now().year - df['year']) * 12 + datetime.datetime.now().month - df['month'])


def func(a, x):
    """
    拟合函数
    """
    k, b = a
    return k * x + b


def dist(a, x, y):
    """
    残差
    """
    return func(a, x) - y


def func_no_b(a, x):
    """
    拟合函数
    """
    k = a
    return k * x


def dist_no_b(a, x, y):
    """
    残差
    """
    return func_no_b(a, x) - y


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


def find_hedge(df, hedge):
    """
    查找保值率
    """
    h = hedge.loc[(hedge['used_years'] == df['used_years']) & (hedge['brand_area'] == df['brand_area']), 'hedge'].values[0]
    return pd.Series([float('%.3f' % (h)), float('%.3f' % (df['price_bn'] * h))])


class FeatureEngineering(object):

    def __init__(self):
        self.train = pd.read_csv(path + '../tmp/train/car_source.csv')
        self.province_city_map = pd.read_csv(path + 'predict/map/province_city_map.csv')
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
        self.open_model_detail = self.open_model_detail.loc[(self.open_model_detail['brand_area'].notnull()), :].reset_index(drop=True)
        self.open_model_detail = self.open_model_detail.rename(columns={'year': 'online_year'})
        self.model_global_mean = self.open_model_detail.loc[:,
                            ['brand_name', 'global_name', 'detail_model', 'brand_slug', 'global_slug', 'model_detail_slug', 'online_year', 'price_bn',
                             'volume', 'control', 'emission_standard', 'brand_area']]
        self.open_model_detail = self.open_model_detail.loc[:,
                            ['brand_name', 'global_name', 'detail_model', 'model_detail_slug', 'online_year',
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

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0
        median_price['rate'] = median_price['median_price'] / median_price['price_bn']
        median_price = median_price.loc[(median_price['rate'] < 1), :].reset_index(drop=True)

        # 拟合车系年份线性k参数
        brand_area_year = median_price.loc[:, ['brand_area', 'used_years']].drop_duplicates(['brand_area', 'used_years']).reset_index(drop=True)

        count = 0
        k_param = pd.DataFrame([], columns=['brand_area', 'used_years', 'k', 'b'])
        for brand_area, used_years in brand_area_year.loc[:, ['brand_area', 'used_years']].values:
            temp = median_price.loc[(median_price['brand_area'] == brand_area) & (median_price['used_years'] == used_years), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            param = [-1, 0]
            var = leastsq(dist, param, args=(np.array(list(temp.price_bn.values)), np.array(list(temp.rate.values))))
            k, b = var[0]
            k_param.loc[count, ['brand_area', 'used_years', 'k', 'b']] = [brand_area, used_years, k, b]
            count = count + 1
        k_param = k_param.sort_values(by=['brand_area', 'used_years']).reset_index(drop=True)

        # 计算牛顿k参数
        count = 0
        newton_k = pd.DataFrame([], columns=['brand_area', 'c', 'k'])
        for brand_area in list(set(k_param.brand_area.values)):
            temp = k_param.loc[(k_param['brand_area'] == brand_area), :].reset_index(drop=True)
            median_b = temp.loc[(k_param['used_years'] == 0), 'b'].values
            if len(median_b) == 0:
                b = median(k_param.loc[(k_param['used_years'] == 0), 'b'].values)
            else:
                b = median_b[0]
            k = cal_newton_min_param(list(k_param.used_years.values), list(k_param.b.values), b)
            newton_k.loc[count, ['brand_area', 'c', 'k']] = [brand_area, b, k]
            count = count + 1

        # 生成车系保值率
        hedge = pd.DataFrame([], columns=['brand_area', 'used_years', 'hedge'])
        for brand_area in list(set(newton_k.brand_area.values)):
            c, k = newton_k.loc[(newton_k['brand_area'] == brand_area), ['c', 'k']].values[0]
            temp = pd.DataFrame([[brand_area, i, c * math.e ** (k * i)] for i in range(0, 21)],columns=['brand_area', 'used_years', 'hedge'])
            hedge = hedge.append(temp)

        self.model_global_mean = self.model_global_mean.merge(median_price.loc[:, ['model_detail_slug', 'median_price']], how='left', on=['model_detail_slug'])
        self.model_global_mean = self.model_global_mean.sort_values(by=['brand_name', 'global_name', 'online_year', 'price_bn']).reset_index(drop=True)
        self.model_global_mean['used_years'] = datetime.datetime.now().year - self.model_global_mean['online_year']
        self.model_global_mean.loc[(self.model_global_mean['used_years'] < 0), 'used_years'] = 0
        self.model_global_mean['hedge'] = np.NAN
        self.model_global_mean['predict_price'] = np.NAN
        self.model_global_mean[['hedge', 'predict_price']] = self.model_global_mean.apply(find_hedge, args=(hedge,), axis=1)
        self.model_global_mean.to_csv(path + '../tmp/train/model_global_mean.csv', index=False)

    def generate_province_div_map(self):
        """
        生成省份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算均值
        median_price = self.train.groupby(['model_detail_slug'])['price'].median().reset_index().rename(columns={'price':'median_value'})
        self.train = self.train.merge(median_price, how='left', on=['model_detail_slug'])
        self.train['price_div'] = (self.train['price'] - self.train['median_value'])
        self.train = self.train.groupby(['province'])['price_div'].median().reset_index()
        miss_province = list(set(self.province_city_map.province.values).difference(set(self.train.province.values)))
        miss_province = pd.DataFrame(pd.Series(miss_province), columns=['province'])
        miss_province['price_div'] = 0
        self.train = self.train.append(miss_province)
        self.train.to_csv(path + '../tmp/train/div_province.csv', index=False)

    def generate_warehouse_years_div_map(self):
        """
        上牌年份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算均值
        self.train = self.train.groupby(by=['brand_slug', 'model_slug', 'model_detail_slug', 'price_bn', 'online_year', 'year'])['price'].median().reset_index().rename(columns={'price': 'median_value'})
        self.train['warehouse_year'] = self.train['year'] - self.train['online_year']

        # 根据年限,统计上牌年份的价格差
        detail_year = self.train.loc[:, ['model_detail_slug', 'online_year']]
        detail_year = detail_year.drop_duplicates(['model_detail_slug', 'online_year']).reset_index(drop=True)

        count = 0
        result = pd.DataFrame([], columns=['model_detail_slug', 'median_value', 'online_year', 'warehouse_year', 'price_div'])
        for model_detail_slug, online_year in detail_year.loc[:, ['model_detail_slug', 'online_year']].values:
            temp = self.train.loc[(self.train['model_detail_slug'] == model_detail_slug) & (self.train['online_year'] == online_year),:].reset_index(drop=True)
            if (len(temp) <= 1) | (len(temp.loc[(temp['warehouse_year'] == 0), :]) == 0):
                continue
            median_value = temp.loc[(temp['warehouse_year'] == 0), 'median_value'].values[0]
            for i in range(0, len(temp)):
                price_div = temp.loc[i, 'median_value'] - median_value
                warehouse_year = temp.loc[i, 'warehouse_year']
                result.loc[count, ['model_detail_slug', 'median_value', 'online_year', 'warehouse_year', 'price_div']] = [model_detail_slug, median_value, online_year, warehouse_year, price_div]
                count = count + 1
        result['rate'] = result['price_div'] / result['median_value']
        param = [0]
        var = leastsq(dist_no_b, param, args=(np.array(list(result.warehouse_year.values)), np.array(list(result.rate.values))))
        k = var[0]
        k = pd.DataFrame([k], columns=['k'])
        k.to_csv(path + '../tmp/train/div_warehouse.csv', index=False)

    def generate_mile_div_map(self):
        """
        公里数差异
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')

        # 根据款型计算均值
        median_value = self.train.groupby(by=['model_detail_slug', 'province', 'year'])['price'].median().reset_index().rename(columns={'price': 'median_value'})
        self.train = self.train.merge(median_value, how='left', on=['model_detail_slug', 'province', 'year'])
        self.train['mile_per_month'] = self.train.apply(cal_use_mile_per_months, axis=1)
        self.train['price_div'] = (self.train['price'] - self.train['median_value']) / self.train['median_value']
        param = [0, 0]
        var = leastsq(dist, param, args=(np.array(list(self.train.mile_per_month.values)), np.array(list(self.train.price_div.values))))
        k, b = var[0]
        k = pd.DataFrame([[k, b]], columns=['k', 'b'])
        k.to_csv(path + '../tmp/train/div_mile.csv', index=False)

    def execute(self):
        """
        执行
        """
        self.handle_data_quality()
        self.handle_data_preprocess()
        self.generate_model_global_mean()
        self.generate_province_div_map()
        self.generate_warehouse_years_div_map()
        self.generate_mile_div_map()