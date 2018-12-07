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


def adjust_k(df):
    """
    调整k参数
    """
    if int(df['k']) == -1:
        return df['hedge']
    else:
        return df['k']


class FeatureEngineering(object):

    def __init__(self):
        self.train = pd.read_csv(path + '../tmp/train/car_source_match.csv')
        self.province_city_map = pd.read_csv(path + 'predict/map/province_city_map.csv')

    ###########################
    # 数据清洗
    ###########################
    def handle_data_quality(self):
        """
        处理数据质量
        """
        # 删掉price<0的记录
        self.train = self.train[self.train['price'] > 0]
        self.train = self.train[self.train['price_bn'] > 0]
        # 删掉不在20年车龄,月份1-12之外记录
        self.train = self.train[(self.train['year'] <= datetime.datetime.now().year)]
        self.train = self.train[self.train['month'].isin(list(np.arange(1, 13)))]
        # 删除掉未知城市,未知款型的记录
        cities = list(set(self.province_city_map.city.values))
        self.train = self.train[self.train['city'].isin(cities)].reset_index(drop=True)
        self.train = self.train.merge(self.province_city_map.loc[:, ['province', 'city']], how='left', on=['city'])
        self.train.reset_index(inplace=True, drop='index')

    def handle_data_preprocess(self):
        """
        数据预处理
        1.删除离群点
        """
        # 删除数据量小于5的款型
        detail_num = self.train.groupby(['detail_slug'])['detail_name'].count().reset_index()
        detail_num = detail_num.loc[(detail_num['detail_name'] >= 5), :]
        self.train = self.train.loc[(self.train['detail_slug'].isin(list(set(detail_num.detail_slug.values)))), :]
        self.train.reset_index(inplace=True, drop=True)

        # 根据款型计算均值
        mean_value = self.train.groupby(['detail_slug'])['price'].mean().reset_index().rename(
            columns={'price': 'mean_value'})
        self.train = self.train.merge(mean_value, how='left', on=['detail_slug'])

        # 根据款型计算标准差
        std_value = self.train.groupby(['detail_slug'])['price'].std().reset_index().rename(
            columns={'price': 'std_value'})
        self.train = self.train.merge(std_value, how='left', on=['detail_slug'])

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

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_name', 'model_name', 'online_year', 'price_bn']).reset_index(drop=True)
        # 取低配数据
        median_price = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :]

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0
        median_price['rate'] = median_price['median_price'] / median_price['price_bn']

        # 拟合车系年份线性k参数
        brand_area_year = median_price.loc[:, ['model_slug', 'model_name', 'used_years']].drop_duplicates(['model_slug', 'model_name', 'used_years']).reset_index(drop=True)

        count = 0
        k_param = pd.DataFrame([], columns=['model_slug', 'model_name', 'used_years', 'k'])
        for model_slug, model_name, used_years in brand_area_year.loc[:, ['model_slug', 'model_name', 'used_years']].values:
            temp = median_price.loc[(median_price['model_slug'] == model_slug) & (median_price['used_years'] == used_years), :].reset_index(drop=True)
            # param = [-1, 0]
            # var = leastsq(dist, param, args=(np.array(list(temp.price_bn.values)), np.array(list(temp.rate.values))))
            # k, b = var[0]
            k = median(list(temp.rate.values))
            k_param.loc[count, ['model_slug', 'model_name', 'used_years', 'k']] = [model_slug, model_name, used_years, k]
            count = count + 1

        # 生成标准车系拟合曲线
        model = k_param.groupby(['model_name'])['used_years'].count().reset_index().sort_values(by=['used_years'])
        model = model.loc[(model['used_years'] >= 5), :].reset_index(drop=True)
        models = k_param.loc[(k_param['model_name'].isin(list(set(model.model_name.values)))), :].reset_index(drop=True)

        count = 0
        k_b_param = pd.DataFrame([], columns=['model_name', 'k', 'b'])
        for model_name in list(set(models.model_name.values)):
            temp = models.loc[(models['model_name'] == model_name), :].reset_index(drop=True)
            param = [-1, 0]
            var = leastsq(dist, param, args=(np.array(list(temp.used_years.values)), np.array(list(temp.k.values))))
            k, b = var[0]
            k_b_param.loc[count, ['model_name', 'k', 'b']] = [model_name, k, b]
            count = count + 1
        k_b_param = k_b_param.sort_values(by=['b']).reset_index(drop=True)
        k_b_param.to_csv(path + '../tmp/train/line_k_param.csv', index=False)

        # # 计算牛顿k参数
        # count = 0
        # newton_k = pd.DataFrame([], columns=['brand_area', 'c', 'k'])
        # for brand_area in list(set(k_param.brand_area.values)):
        #     temp = k_param.loc[(k_param['brand_area'] == brand_area), :].reset_index(drop=True)
        #     median_b = temp.loc[(temp['used_years'] == 0), 'k'].values
        #     if len(median_b) == 0:
        #         b = median(k_param.loc[(k_param['used_years'] == 0), 'k'].values)
        #     else:
        #         b = median_b[0]
        #     k = cal_newton_min_param(list(k_param.used_years.values), list(k_param.k.values), b)
        #     newton_k.loc[count, ['brand_area', 'c', 'k']] = [brand_area, b, k]
        #     count = count + 1
        #
        # # 生成车系保值率
        # hedge = pd.DataFrame([], columns=['brand_area', 'used_years', 'hedge'])
        # for brand_area in list(set(newton_k.brand_area.values)):
        #     c, k = newton_k.loc[(newton_k['brand_area'] == brand_area), ['c', 'k']].values[0]
        #     temp = pd.DataFrame([[brand_area, i, c * math.e ** (k * i)] for i in range(0, 21)],columns=['brand_area', 'used_years', 'hedge'])
        #     hedge = hedge.append(temp).reset_index(drop=True)
        # hedge = hedge.merge(k_param, how='left', on=['brand_area', 'used_years'])
        # hedge['k'] = hedge['k'].fillna(-1)
        # hedge['hedge'] = hedge.apply(adjust_k, axis=1)
        # hedge = hedge.drop(['k'], axis=1)
        # hedge.to_csv(path + '../tmp/train/hedge.csv', index=False)

        # # 生成全局均值
        # self.model_global_mean = self.model_global_mean.merge(median_price.loc[:, ['model_detail_slug', 'median_price']], how='left', on=['model_detail_slug'])
        # self.model_global_mean = self.model_global_mean.sort_values(by=['brand_name', 'global_name', 'online_year', 'price_bn']).reset_index(drop=True)
        # self.model_global_mean['used_years'] = datetime.datetime.now().year - self.model_global_mean['online_year']
        # self.model_global_mean.loc[(self.model_global_mean['used_years'] < 0), 'used_years'] = 0
        # self.model_global_mean['hedge'] = np.NAN
        # self.model_global_mean['predict_price'] = np.NAN
        # self.model_global_mean[['hedge', 'predict_price']] = self.model_global_mean.apply(find_hedge, args=(hedge,), axis=1)
        # self.model_global_mean = self.model_global_mean.sort_values(by=['brand_slug', 'global_slug', 'online_year', 'price_bn'],ascending=[True, True, False, True]).reset_index(drop=True)
        # final = pd.DataFrame()
        # for global_slug in list(set(self.model_global_mean.global_slug.values)):
        #     temp = self.model_global_mean.loc[(self.model_global_mean['global_slug'] == global_slug), :].reset_index(drop=True)
        #     if len(temp.loc[(temp['median_price'].notnull()), :]) == 0:
        #         temp['final_price'] = temp['predict_price']
        #         final = final.append(temp, sort=False).reset_index(drop=True)
        #     else:
        #         m_price = temp.loc[(temp['median_price'].notnull()), 'median_price'].values[0]
        #         p_price = temp.loc[(temp['median_price'].notnull()), 'predict_price'].values[0]
        #         rate = m_price / p_price
        #         temp['final_price'] = temp['predict_price'] * rate
        #         final = final.append(temp, sort=False).reset_index(drop=True)
        # final = final.sort_values(by=['brand_slug', 'global_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        # final.to_csv(path + '../tmp/train/model_global_mean.csv', index=False)

    def generate_price_bn_div_map(self):
        """
        生成指导价差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_area', 'brand_slug', 'model_slug', 'model_detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0

        # 根据年限,统计指导价差值的价格差
        model_year = median_price.loc[:, ['brand_area', 'model_slug', 'used_years']]
        model_year = model_year.drop_duplicates(['brand_area', 'model_slug', 'used_years']).reset_index(drop=True)

        count = 0
        result = pd.DataFrame([], columns=['brand_area', 'model_slug', 'used_years', 'price_bn_div', 'price_div'])
        for brand_area, model_slug, used_years in model_year.loc[:, ['brand_area', 'model_slug', 'used_years']].values:
            temp = median_price.loc[(median_price['model_slug'] == model_slug) & (median_price['used_years'] == used_years), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            for i in range(1, len(temp)):
                price_bn_div = temp.loc[i, 'price_bn'] - temp.loc[0, 'price_bn']
                price_div = temp.loc[i, 'median_price'] - temp.loc[0, 'median_price']
                result.loc[count, ['brand_area', 'model_slug', 'used_years', 'price_bn_div', 'price_div']] = [brand_area, model_slug, used_years, price_bn_div, price_div]
                count = count + 1

        brand_area_year = result.drop_duplicates(['brand_area', 'used_years']).reset_index(drop=True)
        count = 0
        k_param = pd.DataFrame([], columns=['brand_area', 'used_years', 'k'])
        for brand_area, used_years in brand_area_year.loc[:, ['brand_area', 'used_years']].values:
            temp = result.loc[(result['brand_area'] == brand_area) & (result['used_years'] == used_years), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            param = [0]
            var = leastsq(dist_no_b, param, args=(np.array(list(temp.price_bn_div.values)), np.array(list(temp.price_div.values))))
            k = var[0][0]
            k_param.loc[count, ['brand_area', 'used_years', 'k']] = [brand_area, used_years, k]
            count = count + 1
        k_param = k_param.sort_values(by=['brand_area', 'used_years']).reset_index(drop=True)

        # 计算牛顿k参数
        count = 0
        newton_k = pd.DataFrame([], columns=['brand_area', 'c', 'k'])
        for brand_area in list(set(k_param.brand_area.values)):
            temp = k_param.loc[(k_param['brand_area'] == brand_area), :].reset_index(drop=True)
            median_b = temp.loc[(temp['used_years'] == 0), 'k'].values
            if len(median_b) == 0:
                b = median(k_param.loc[(k_param['used_years'] == 0), 'k'].values)
            elif median_b[0] < 0.7:
                b = 0.8
            else:
                b = median_b[0]
            k = cal_newton_min_param(list(k_param.used_years.values), list(k_param.k.values), b)
            newton_k.loc[count, ['brand_area', 'c', 'k']] = [brand_area, b, k]
            count = count + 1

        # 生成指导价衰减表
        hedge = pd.DataFrame([], columns=['brand_area', 'used_years', 'hedge'])
        for brand_area in list(set(newton_k.brand_area.values)):
            c, k = newton_k.loc[(newton_k['brand_area'] == brand_area), ['c', 'k']].values[0]
            temp = pd.DataFrame([[brand_area, i, c * math.e ** (k * i)] for i in range(0, 21)], columns=['brand_area', 'used_years', 'hedge'])
            hedge = hedge.append(temp).reset_index(drop=True)
        hedge = hedge.merge(k_param, how='left', on=['brand_area', 'used_years'])
        hedge['k'] = hedge['k'].fillna(-1)
        hedge['hedge'] = hedge.apply(adjust_k, axis=1)
        hedge = hedge.drop(['k'], axis=1)
        hedge.to_csv(path + '../tmp/train/div_price_bn.csv', index=False)

    def generate_province_div_map(self):
        """
        生成省份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算均值
        median_price = self.train.groupby(['model_detail_slug'])['price'].median().reset_index().rename(columns={'price':'median_value'})
        self.train = self.train.merge(median_price, how='left', on=['model_detail_slug'])
        self.train['price_div'] = (self.train['price'] - self.train['median_value'])
        self.train = self.train.groupby(['brand_area', 'province'])['price_div'].median().reset_index()
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
        # self.generate_price_bn_div_map()
        # self.generate_province_div_map()
        # self.generate_warehouse_years_div_map()
        # self.generate_mile_div_map()