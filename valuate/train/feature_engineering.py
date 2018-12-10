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
    used_months = ((datetime.datetime.now().year - df['year']) * 12 + datetime.datetime.now().month - df['month'])
    if used_months <= 0:
        used_months = 1
    return df['mile'] / used_months


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


def find_best_k_b(origin, k_b_param):
    """
    查找最佳kb参数
    """
    param = k_b_param.copy()
    param['distance'] = np.NAN
    for i in range(0, len(param)):
        k, b = param.loc[i, ['k', 'b']].values
        distance = sum([(k*used_years+b-rate)**2 for used_years, rate in zip(list(origin.used_years.values), list(origin.rate.values))])
        param.loc[i, ['distance']] = distance
    # 取距离最低kb
    param = param.sort_values(by=['distance']).reset_index(drop=True)
    return param.loc[0, ['k', 'b']].values


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
        self.car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_all.csv')
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

    def generate_model_map(self):
        """
        生成全款型全国均值模型
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        # 取低配数据
        median_price = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :]

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0
        median_price['rate'] = median_price['median_price'] / median_price['price_bn']

        # 生成标准车系拟合曲线
        model = median_price.groupby(['model_slug'])['used_years'].count().reset_index().sort_values(by=['used_years'])
        have_data_model = model.loc[(model['used_years'] >= 5), :].reset_index(drop=True)
        models = median_price.loc[(median_price['model_slug'].isin(list(set(have_data_model.model_slug.values)))), :].reset_index(drop=True)

        count = 0
        k_b_param = pd.DataFrame([], columns=['model_slug', 'k', 'b'])
        for model_slug in list(set(models.model_slug.values)):
            temp = models.loc[(models['model_slug'] == model_slug), :].reset_index(drop=True)
            param = [-1, 0]
            var = leastsq(dist, param, args=(np.array(list(temp.used_years.values)), np.array(list(temp.rate.values))))
            k, b = var[0]
            k_b_param.loc[count, ['model_slug', 'k', 'b']] = [model_slug, k, b]
            count = count + 1
        k_b_param = k_b_param.sort_values(by=['b']).reset_index(drop=True)
        k_b_param['model_slug'] = k_b_param['model_slug'].astype(int)
        k_b_param['step'] = 0

        # 缺失车系阶段1,查找已有车系接近kb
        lack_data_model = model.loc[(model['used_years'] < 5), :].reset_index(drop=True)
        models = median_price.loc[(median_price['model_slug'].isin(list(set(lack_data_model.model_slug.values)))), :].reset_index(drop=True)

        count = 0
        step1 = pd.DataFrame([], columns=['model_slug', 'k', 'b'])
        for model_slug in list(set(models.model_slug.values)):
            temp = models.loc[(models['model_slug'] == model_slug), :].reset_index(drop=True)
            k, b = find_best_k_b(temp, k_b_param)
            step1.loc[count, ['model_slug', 'k', 'b']] = [model_slug, k, b]
            count = count + 1
        step1['model_slug'] = step1['model_slug'].astype(int)
        step1['step'] = 1
        k_b_param = k_b_param.append(step1, sort=False).reset_index(drop=True)
        k_b_param = k_b_param.merge(self.car_autohome_all.loc[:, ['brand_slug', 'model_slug']].drop_duplicates(['model_slug']), how='left', on=['model_slug'])

        # 缺失车系阶段2,查找已有品牌接近kb
        lack_models = self.car_autohome_all.loc[~(self.car_autohome_all['model_slug'].isin(list(set(k_b_param.model_slug.values)))), ['brand_slug', 'model_slug']].drop_duplicates(['model_slug']).reset_index(drop=True)
        count = 0
        step2 = pd.DataFrame([], columns=['brand_slug', 'model_slug', 'k', 'b'])
        for i in range(0, len(lack_models)):
            brand_slug, model_slug = lack_models.loc[i, ['brand_slug', 'model_slug']].values
            temp = k_b_param.loc[(k_b_param['brand_slug'] == brand_slug), :].sort_values(by=['b']).reset_index(drop=True)
            if len(temp) == 0:
                continue
            k, b = temp.loc[len(temp) % 2 + int(len(temp)/2) - 1, ['k', 'b']].values
            step2.loc[count, ['brand_slug', 'model_slug', 'k', 'b']] = [brand_slug, model_slug, k, b]
            count = count + 1
        step2['model_slug'] = step2['model_slug'].astype(int)
        step2['step'] = 2
        k_b_param = k_b_param.append(step2, sort=False).reset_index(drop=True)

        # 缺失品牌车系阶段3,临时
        lack_models = self.car_autohome_all.loc[~(self.car_autohome_all['model_slug'].isin(list(set(k_b_param.model_slug.values)))), ['brand_slug', 'model_slug']].drop_duplicates(['model_slug']).reset_index(drop=True)
        count = 0
        step3 = pd.DataFrame([], columns=['brand_slug', 'model_slug', 'k', 'b'])
        for i in range(0, len(lack_models)):
            brand_slug, model_slug = lack_models.loc[i, ['brand_slug', 'model_slug']].values
            k, b = k_b_param.loc[len(k_b_param) % 2 + int(len(k_b_param) / 2) - 1, ['k', 'b']].values
            step3.loc[count, ['brand_slug', 'model_slug', 'k', 'b']] = [brand_slug, model_slug, k, b]
            count = count + 1
        step3['model_slug'] = step3['model_slug'].astype(int)
        step3['step'] = 3
        k_b_param = k_b_param.append(step3, sort=False).reset_index(drop=True)

        k_b_param = k_b_param.merge(self.car_autohome_all.loc[:, ['brand_name', 'model_slug', 'model_name', 'body', 'energy']].drop_duplicates(['model_slug']), how='left', on=['model_slug'])
        k_b_param.to_csv(path + '../tmp/train/model_k_param.csv', index=False)

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

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0

        # 根据年限,统计指导价差值的价格差
        model_year = median_price.loc[:, ['brand_slug', 'model_slug', 'used_years']]
        model_year = model_year.drop_duplicates(['brand_slug', 'model_slug', 'used_years']).reset_index(drop=True)

        count = 0
        result = pd.DataFrame([], columns=['brand_slug', 'model_slug', 'used_years', 'price_bn_div', 'price_div'])
        for brand_slug, model_slug, used_years in model_year.loc[:, ['brand_slug', 'model_slug', 'used_years']].values:
            temp = median_price.loc[(median_price['model_slug'] == model_slug) & (median_price['used_years'] == used_years), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            for i in range(1, len(temp)):
                price_bn_div = temp.loc[i, 'price_bn'] - temp.loc[0, 'price_bn']
                price_div = temp.loc[i, 'median_price'] - temp.loc[0, 'median_price']
                result.loc[count, ['brand_slug', 'model_slug', 'used_years', 'price_bn_div', 'price_div']] = [brand_slug, model_slug, used_years, price_bn_div, price_div]
                count = count + 1
        result = result.loc[(result['price_bn_div'] <= 100) & (result['price_div'] > 0), :].reset_index(drop=True)
        result.to_csv(path + '../tmp/train/div_price_bn_data.csv', index=False)

        count = 0
        k_param = pd.DataFrame([], columns=['used_years', 'k'])
        for used_years in list(set(result.used_years.values)):
            temp = result.loc[(result['used_years'] == used_years), :].reset_index(drop=True)
            if len(temp) <= 1:
                continue
            param = [0]
            var = leastsq(dist_no_b, param, args=(np.array(list(temp.price_bn_div.values)), np.array(list(temp.price_div.values))))
            k = var[0][0]
            k_param.loc[count, ['used_years', 'k']] = [used_years, k]
            count = count + 1
        k_param = k_param.sort_values(by=['used_years']).reset_index(drop=True)
        k_param.to_csv(path + '../tmp/train/div_price_bn_k_param.csv', index=False)

    def generate_province_div_map(self):
        """
        生成省份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        # 取低配数据
        median_price = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :]

        median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price.loc[(median_price['used_years'] < 0), 'used_years'] = 0

        # 根据款型计算均值
        self.train = self.train.loc[(self.train['detail_slug'].isin(list(set(median_price.detail_slug.values)))), :].reset_index(drop=True)
        self.train = self.train.merge(median_price.loc[:, ['detail_slug', 'used_years', 'median_price']], how='left', on=['detail_slug'])
        self.train['price_div'] = (self.train['price'] - self.train['median_price'])
        self.train = self.train.loc[(self.train['price'] < 100), :].reset_index(drop=True)
        self.train.to_csv(path + '../tmp/train/div_province_data.csv', index=False)

        # GDP越高的城市更偏爱豪车
        count = 0
        k_param = pd.DataFrame([], columns=['province', 'k', 'b'])
        for province in list(set(self.train.province.values)):
            temp = self.train.loc[(self.train['province'] == province), :].reset_index(drop=True)
            param = [0, 0]
            var = leastsq(dist, param, args=(np.array(list(temp.price.values)), np.array(list(temp.price_div.values))))
            k, b = var[0]
            k_param.loc[count, ['province', 'k', 'b']] = [province, k, b]
            count = count + 1

        miss_province = list(set(self.province_city_map.province.values).difference(set(k_param.province.values)))
        miss_province = pd.DataFrame(pd.Series(miss_province), columns=['province'])
        miss_province['k'] = 0
        miss_province['b'] = 0
        k_param = k_param.append(miss_province, sort=False)
        k_param.to_csv(path + '../tmp/train/div_province_k_param.csv', index=False)

    def generate_warehouse_years_div_map(self):
        """
        上牌年份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn', 'year'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        median_price['warehouse_year'] = median_price['year'] - median_price['online_year']

        # 根据年限,统计上牌年份的价格差
        detail_year = median_price.loc[:, ['detail_slug', 'online_year']]
        detail_year = detail_year.drop_duplicates(['detail_slug', 'online_year']).reset_index(drop=True)

        count = 0
        result = pd.DataFrame([], columns=['detail_slug', 'median_price', 'online_year', 'warehouse_year', 'price_div'])
        for detail_slug, online_year in detail_year.loc[:, ['detail_slug', 'online_year']].values:
            temp = median_price.loc[(median_price['detail_slug'] == detail_slug) & (median_price['online_year'] == online_year), :].reset_index(drop=True)
            if (len(temp) <= 1) | (len(temp.loc[(temp['warehouse_year'] == 0), :]) == 0):
                continue
            temp_median_price = temp.loc[(temp['warehouse_year'] == 0), 'median_price'].values[0]
            for i in range(0, len(temp)):
                price_div = temp.loc[i, 'median_price'] - temp_median_price
                warehouse_year = temp.loc[i, 'warehouse_year']
                result.loc[count, ['detail_slug', 'median_price', 'online_year', 'warehouse_year', 'price_div']] = [detail_slug, temp_median_price, online_year, warehouse_year, price_div]
                count = count + 1
        result['rate'] = result['price_div'] / result['median_price']
        result.to_csv(path + '../tmp/train/div_warehouse_data.csv', index=False)
        param = [0]
        var = leastsq(dist_no_b, param, args=(np.array(list(result.warehouse_year.values)), np.array(list(result.rate.values))))
        k = var[0][0]
        k = pd.DataFrame([k], columns=['k'])
        k.to_csv(path + '../tmp/train/div_warehouse_k_param.csv', index=False)

    def generate_mile_div_map(self):
        """
        公里数差异
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算均值
        median_price = self.train.groupby(by=['detail_slug', 'province', 'online_year'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        self.train = self.train.merge(median_price, how='left', on=['detail_slug', 'province', 'online_year'])
        self.train['mile_per_month'] = self.train.apply(cal_use_mile_per_months, axis=1)
        self.train['rate'] = (self.train['price'] - self.train['median_price']) / self.train['median_price']
        param = [0, 0]
        var = leastsq(dist, param, args=(np.array(list(self.train.mile_per_month.values)), np.array(list(self.train.rate.values))))
        k, b = var[0]
        k = pd.DataFrame([[k, b]], columns=['k', 'b'])
        k.to_csv(path + '../tmp/train/div_mile_k_param.csv', index=False)

    def execute(self):
        """
        执行
        """
        self.handle_data_quality()
        self.handle_data_preprocess()
        self.generate_model_map()
        self.generate_price_bn_div_map()
        self.generate_province_div_map()
        self.generate_warehouse_years_div_map()
        self.generate_mile_div_map()