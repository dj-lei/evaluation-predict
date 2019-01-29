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
        self.car_autohome_all = self.car_autohome_all.drop(['volume', 'control', 'volume_extend', 'emission_standard'], axis=1)
        self.province_city_map = pd.read_csv(path + '../tmp/train/province_city_map.csv')

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

        # median_price['used_years'] = datetime.datetime.now().year - median_price['online_year']
        median_price['used_years'] = 2018 - median_price['online_year']
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

    def generate_model_map(self):
        """
        生成全款型全国均值模型
        """
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
        div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param.csv')
        car_autohome_all = self.car_autohome_all.copy()
        car_autohome_all = car_autohome_all.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        car_autohome_all['used_years'] = datetime.datetime.now().year - car_autohome_all['online_year']
        car_autohome_all.loc[(car_autohome_all['used_years'] < 0), 'used_years'] = 0

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        median_price.to_csv(path + '../tmp/train/model_data.csv', index=False)

        # 取低配数据
        low_config_car = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
        low_config_car = low_config_car.drop_duplicates(['model_slug', 'online_year'])

        # 调整指导价差,确保同条件下高配比低配价格高
        part1 = pd.DataFrame()
        for model_slug, online_year in low_config_car.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug)&(car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(median_price.loc[:, ['detail_slug', 'median_price']], how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            part1 = part1.append(car_autohome_temp, sort=False).reset_index(drop=True)
        part1.to_csv(path + '../tmp/train/global_model_mean_part1.csv', index=False)

        content = input("是否重新生成人工评估文件:")
        if content == 'y':
            # 需要人工评估的低配车
            part2 = car_autohome_all.merge(part1.loc[:, ['detail_slug', 'median_price']], how='left', on=['detail_slug'])
            part2 = part2.loc[(part2['median_price'].isnull()), :].reset_index(drop=True)
            # 取低配数据
            part2 = part2.loc[part2.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
            part2.to_csv(path + '../tmp/train/global_model_mean_part2.csv', index=False)

    def generate_manual_model_map(self):
        """
        组合人工评估数据
        """
        part1 = pd.read_csv(path + '../tmp/train/global_model_mean_part1.csv')
        low_config_car = pd.read_csv(path + '../tmp/train/global_model_mean_part2.csv')
        low_config_car['median_price'] = low_config_car['median_price'] / 0.95
        div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param.csv')
        combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv')

        car_autohome_all = self.car_autohome_all.copy()
        car_autohome_all = car_autohome_all.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        car_autohome_all['used_years'] = datetime.datetime.now().year - car_autohome_all['online_year']
        car_autohome_all.loc[(car_autohome_all['used_years'] < 0), 'used_years'] = 0

        low_config_car = low_config_car.loc[(low_config_car['detail_slug'].isin(list(car_autohome_all.detail_slug.values))), :]

        # 调整指导价差,确保同条件下高配比低配价格高
        part2 = pd.DataFrame()
        for model_slug, online_year in low_config_car.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug)&(car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(low_config_car.loc[:, ['detail_slug', 'median_price']], how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            part2 = part2.append(car_autohome_temp, sort=False).reset_index(drop=True)

        global_model_mean = part1.append(part2, sort=False).reset_index(drop=True)
        print('最终组合款型数:', len(global_model_mean))
        global_model_mean = global_model_mean.merge(combine_detail.loc[:, ['detail_model_slug', 'car_autohome_detail_id']].rename(columns={'car_autohome_detail_id':'detail_slug'}), how='left', on=['detail_slug'])
        global_model_mean = global_model_mean.loc[(global_model_mean['detail_model_slug'].notnull()), :]
        print('最终组合款型数:', len(global_model_mean))
        global_model_mean.to_csv(path + '../tmp/train/global_model_mean.csv', index=False)

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
        k_param = k_param.merge(self.province_city_map.loc[:, ['province', 'city']], how='left', on=['province'])
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
        self.train = self.train.loc[(self.train['mile_per_month'] < 1), :].reset_index(drop=True)
        self.train.to_csv(path + '../tmp/train/div_mile_data.csv', index=False)
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
        self.generate_price_bn_div_map()
        # self.generate_model_map()
        # self.generate_manual_model_map()
        self.generate_warehouse_years_div_map()
        # self.generate_province_div_map()
        # self.generate_mile_div_map()

