from evaluation.train import *


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


def cal_z_score(df):
    """
    计算z得分
    """
    if df['std_value'] <= 0:
        return 0
    return abs((df['price'] - df['mean_value']) / (df['std_value']))


def update_price(df, low_config_car):
    """
    更新价格
    """
    price = low_config_car.loc[(low_config_car['detail_slug'] == df['detail_slug']),'median_price'].values
    if len(price) == 0:
        return pd.Series([df['median_price'], df['update_time']])
    else:
        return pd.Series([price[0], datetime.datetime.now().strftime("%Y-%m-%d")])


def cal_profit_rate(df):
    temp = df['adjust_sell_price']
    for i in range(0, 10000):
        buy = temp + i * 0.03
        rate = 0.34 * math.e ** (-0.6 * math.log(buy, math.e))
        if rate <= 0.101:
            rate = 0.101
        div = df['adjust_sell_price'] - buy * (1 - rate)
        if div > 0:
            continue
        else:
            return buy


def adjust_condition(df):
    # 车况判断两年以内优秀,8-3年良好,9-11年一般,12年以上较差
    used_years = df['used_years']
    if used_years <= 2:
        condition = 'excellent'
    elif 2 < used_years <= 8:
        condition = 'good'
    elif 8 < used_years <= 11:
        condition = 'fair'
    elif 11 < used_years:
        condition = 'bad'

    if condition != df['condition']:
        return float('%.2f' % (
                    (df['adjust_buy_price'] / gl.CAR_CONDITION_COEFFICIENT[df['condition']]) * gl.CAR_CONDITION_COEFFICIENT[condition]))
    return float('%.2f' % (df['adjust_buy_price']))


def update_price_ttp(df, part):
    price = part.loc[(part['detail_slug'] == df['detail_slug']), ['median_price']].values
    if len(price) != 0:
        return pd.Series([price[0][0], datetime.datetime.now().strftime("%Y-%m-%d")])
    return pd.Series([df['median_price'], df['update_time']])


def adjust_all_price(df):
    """
    调整整体
    """
    if df['all_median_price'] < df['median_price']:
        return df['all_median_price']
    return df['median_price']


def is_adjust(df):
    if abs(df['median_price'] - df['private_median_price']) <= 0.5:
        return 1
    elif (abs(df['private_median_price']-df['median_price'])/df['private_median_price']) <= 0.15:
        return 1
    return 0


class FeatureEngineeringCron(object):

    def __init__(self, control):
        self.control = control

        car_source = pd.read_csv(path + '../tmp/train/train.csv')
        self.train = car_source.loc[(car_source['type'] == 'personal') & (car_source['control'] == self.control), :].reset_index(drop=True)
        self.sell = car_source.loc[(car_source['type'] == 'sell') & (car_source['control'] == self.control), :].reset_index(drop=True)
        self.car_autohome_all = pd.read_csv(path + '../tmp/train/car_autohome_all.csv')
        self.car_autohome_all = self.car_autohome_all.loc[(self.car_autohome_all['control'] == self.control), :].reset_index(drop=True)
        self.car_autohome_all = self.car_autohome_all.drop(['volume', 'volume_extend', 'emission_standard'], axis=1)
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
        mean_value = self.train.groupby(['detail_slug'])['price'].mean().reset_index().rename(columns={'price': 'mean_value'})
        self.train = self.train.merge(mean_value, how='left', on=['detail_slug'])

        # 根据款型计算标准差
        std_value = self.train.groupby(['detail_slug'])['price'].std().reset_index().rename(columns={'price': 'std_value'})
        self.train = self.train.merge(std_value, how='left', on=['detail_slug'])

        # 计算Z得分，根据阈值删除离群点
        self.train['z_score'] = self.train.apply(cal_z_score, axis=1)
        self.train = self.train.loc[(self.train['z_score'] <= 1.5), :]
        self.train.reset_index(inplace=True, drop=True)
        self.train = self.train.drop(['mean_value', 'std_value', 'z_score'], axis=1)
        self.train.to_csv(path + '../tmp/train/train_temp_' + self.control + '.csv', index=False)

    def update_price_bn_div_map(self):
        """
        更新指导价差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train_temp_' + self.control + '.csv')

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
        k_param.to_csv(path + '../tmp/train/div_price_bn_k_param_' + self.control + '.csv', index=False)

    def update_warehouse_years_div_map(self):
        """
        上牌年份差异表
        """
        self.train = pd.read_csv(path + '../tmp/train/train_temp_' + self.control + '.csv')
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

        param = [0]
        var = leastsq(dist_no_b, param, args=(np.array(list(result.warehouse_year.values)), np.array(list(result.rate.values))))
        k = var[0][0]
        k = pd.DataFrame([k], columns=['k'])
        k.to_csv(path + '../tmp/train/div_warehouse_k_param_' + self.control + '.csv', index=False)

    def update_model_map(self):
        """
        生成全款型全国均值模型
        """
        self.train = pd.read_csv(path + '../tmp/train/train_temp_' + self.control + '.csv')
        global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean.csv')
        global_model_mean = global_model_mean.loc[(global_model_mean['control'] == self.control), :].reset_index(drop=True)
        global_model_mean = global_model_mean.loc[global_model_mean.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
        div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param_' + self.control + '.csv')
        car_autohome_all = self.car_autohome_all.copy()

        car_autohome_all = car_autohome_all.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        car_autohome_all['used_years'] = datetime.datetime.now().year - car_autohome_all['online_year']
        car_autohome_all.loc[(car_autohome_all['used_years'] < 0), 'used_years'] = 0

        # 整体价格
        all_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'all_median_price'})

        # 上牌时间和上市时间相同
        self.train = self.train.loc[(self.train['online_year'] == self.train['year']), :].reset_index(drop=True)

        # 根据款型计算中位数
        median_price = self.train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        # 整体小于同龄的剔除
        median_price = median_price.merge(all_price.loc[:, ['detail_slug','all_median_price']], how='left', on=['detail_slug'])
        median_price['median_price'] = median_price.apply(adjust_all_price,axis=1)
        median_price = median_price.drop(['all_median_price'], axis=1)

        # 取低配数据
        low_config_car = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).median_price.idxmin(), :].reset_index(drop=True)
        model_year = low_config_car.drop_duplicates(['model_slug', 'online_year'])

        # 调整指导价差,确保同条件下高配比低配价格高
        part1 = pd.DataFrame()
        for model_slug, online_year in model_year.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug) & (car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(low_config_car.loc[:, ['detail_slug', 'median_price']],how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            part1 = part1.append(car_autohome_temp, sort=False).reset_index(drop=True)

        low_config_car = part1.loc[part1.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
        global_model_mean[['median_price', 'update_time']] = global_model_mean.apply(update_price, args=(low_config_car,), axis=1)

        # 调整指导价差,确保同条件下高配比低配价格高
        result = pd.DataFrame()
        for model_slug, online_year, update_time in global_model_mean.loc[:, ['model_slug', 'online_year', 'update_time']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug) & (car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(global_model_mean.loc[:, ['detail_slug', 'median_price']],how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            car_autohome_temp['update_time'] = update_time
            result = result.append(car_autohome_temp, sort=False).reset_index(drop=True)
        result = result.drop_duplicates(['detail_slug'])

        # combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv', low_memory=False)
        # result = result.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        # result = result.merge(combine_detail.loc[:, ['detail_model_slug', 'car_autohome_detail_id']].rename(columns={'car_autohome_detail_id': 'detail_slug'}), how='left', on=['detail_slug'])
        result['step1_price'] = result['median_price']
        result.to_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv', index=False)

    def update_retain_high_config(self):
        """
        保留高配价格
        """
        def retain_high_config(df):
            normal_update_price = median_price.loc[(median_price['detail_slug'] == df['detail_slug']), 'median_price'].values
            if len(normal_update_price) == 0:
                return df['median_price']

            all_update_price = np.median(origin_train.loc[(origin_train['detail_slug'] == df['detail_slug']), 'price'])
            if normal_update_price[0] >= all_update_price:
                update_price = all_update_price
            else:
                update_price = normal_update_price[0]
            return update_price

        combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv', low_memory=False)
        origin_train = pd.read_csv(path + '../tmp/train/train_temp_' + self.control + '.csv')
        global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv')
        # div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param.csv')

        # 上牌时间和上市时间相同
        train = origin_train.copy()
        train = train.loc[(train['online_year'] == train['year']), :].reset_index(drop=True)
        # 根据款型计算中位数
        median_price = train.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'online_year', 'price_bn'])['price'].median().reset_index().rename(columns={'price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)

        # 保留高配价格
        retain_high_config_price = global_model_mean_temp.copy()
        retain_high_config_price['median_price'] = retain_high_config_price.apply(retain_high_config, axis=1)
        model_online = retain_high_config_price.drop_duplicates(['model_slug', 'online_year'])

        # 调整指导价差,确保同条件下高配比低配价格高
        # result_retain = pd.DataFrame()
        # for model_slug, online_year in model_online.loc[:, ['model_slug', 'online_year']].values:
        #     car_autohome_temp = retain_high_config_price.loc[(retain_high_config_price['model_slug'] == model_slug) & (retain_high_config_price['online_year'] == online_year), :].reset_index(drop=True)
        #     used_years = car_autohome_temp.loc[0, 'used_years']
        #     k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
        #     for i in range(0, len(car_autohome_temp) - 1):
        #         if car_autohome_temp.loc[i, 'median_price'] > car_autohome_temp.loc[i + 1, 'median_price']:
        #             div_price_bn = car_autohome_temp.loc[i + 1, 'price_bn'] - car_autohome_temp.loc[i, 'price_bn']
        #             car_autohome_temp.loc[i + 1, 'median_price'] = float('%.3f' % (div_price_bn * k + car_autohome_temp.loc[i, 'median_price']))
        #     result_retain = result_retain.append(car_autohome_temp, sort=False).reset_index(drop=True)

        final = retain_high_config_price.copy()
        final = final.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        final = final.merge(combine_detail.loc[:, ['detail_model_slug', 'car_autohome_detail_id']].rename(columns={'car_autohome_detail_id': 'detail_slug'}), how='left', on=['detail_slug'])
        final['step2_price'] = final['median_price']
        final.to_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv', index=False)
        print('最终组合款型数:', len(final))

    def ttp_reverse_examine(self):
        """
        天天拍车源反向校验
        """
        div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param_' + self.control + '.csv')
        global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv').rename(columns={'median_price': 'private_median_price'})
        wait_reverse = pd.read_csv(path + '../tmp/train/wait_reverse.csv')
        wait_reverse = wait_reverse.loc[(wait_reverse['control'] == self.control), :].reset_index(drop=True)

        car_autohome_all = self.car_autohome_all.copy()
        car_autohome_all = car_autohome_all.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        car_autohome_all['used_years'] = datetime.datetime.now().year - car_autohome_all['online_year']
        car_autohome_all.loc[(car_autohome_all['used_years'] < 0), 'used_years'] = 0

        # global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv').rename(columns={'detail_model_slug': 'gpj_detail_slug'})
        # tiantianpai = pd.read_csv(path + '../script/man.csv')
        # tiantianpai = tiantianpai.loc[(tiantianpai['control'] == self.control), :].reset_index(drop=True)
        # tiantianpai = tiantianpai.loc[:, ['title', 'year', 'month', 'mile', 'condition', 'city', 'price', 'gpj_detail_slug', 'brand_name','model_name', 'detail_name', 'online_year', 'new_sell_price', 'new_buy_price']]
        # tiantianpai = tiantianpai.merge(global_model_mean_temp.loc[:, ['gpj_detail_slug', 'brand_slug', 'model_slug', 'price_bn', 'detail_slug']],how='left', on=['gpj_detail_slug'])

        tiantianpai = self.sell.copy()
        tiantianpai = tiantianpai.append(wait_reverse, sort=False)
        tiantianpai.to_csv(path + '../tmp/train/man.csv', index=False)
        tiantianpai['used_years'] = datetime.datetime.now().year - tiantianpai['online_year']
        tiantianpai.loc[(tiantianpai['used_years'] < 0), 'used_years'] = 0

        # 调整车龄差异
        if self.control == '自动':
            k = 0.055
        else:
            k = 0.08
        tiantianpai['warehouse_year'] = tiantianpai['year'] - tiantianpai['online_year']
        tiantianpai['adjust_sell_price'] = tiantianpai['price'] / (k * tiantianpai['warehouse_year'] + 1)

        # 调整收购价和零售价
        tiantianpai['adjust_buy_price'] = tiantianpai.apply(cal_profit_rate, axis=1)
        # 调整车况
        # tiantianpai['adjust_buy_price'] = tiantianpai.apply(adjust_condition, axis=1)

        # 根据款型计算中位数
        median_price = tiantianpai.groupby(['brand_slug', 'brand_name', 'model_slug', 'model_name', 'detail_slug', 'detail_name', 'online_year', 'price_bn'])['adjust_buy_price'].median().reset_index().rename(columns={'adjust_buy_price': 'median_price'})
        median_price = median_price.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)

        # 相对于个人车源筛选
        median_price = median_price.merge(global_model_mean_temp.loc[:, ['detail_slug', 'private_median_price']],how='left', on=['detail_slug'])
        median_price['is_adjust'] = median_price.apply(is_adjust, axis=1)
        div = median_price.loc[(median_price['is_adjust'] == 0), :].reset_index(drop=True)
        div.to_csv(path + '../tmp/train/差异较大对比_' + self.control + '.csv', index=False)
        median_price = median_price.loc[(median_price['is_adjust'] == 1), :].reset_index(drop=True)

        # 取低配数据
        low_config_car = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(),:].reset_index(drop=True)
        low_config_car = low_config_car.drop_duplicates(['model_slug', 'online_year']).reset_index(drop=True)

        # 调整指导价差,确保同条件下高配比低配价格高
        part1 = pd.DataFrame()
        for model_slug, online_year in low_config_car.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug) & (car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(median_price.loc[:, ['detail_slug', 'median_price']],how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            car_autohome_temp['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d")
            part1 = part1.append(car_autohome_temp, sort=False).reset_index(drop=True)

        global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv')
        global_model_mean_temp[['median_price', 'update_time']] = global_model_mean_temp.apply(update_price_ttp, args=(part1,), axis=1)

        global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean.csv')
        global_model_mean = global_model_mean.loc[(global_model_mean['control'] == self.control), :].reset_index(drop=True)

        global_model_mean_temp['step3_price'] = global_model_mean_temp['median_price']
        global_model_mean_temp = global_model_mean_temp.merge(global_model_mean.loc[:,['detail_slug','median_price']].rename(columns={'median_price':'last_verison_price'}),how='left',on=['detail_slug'])
        global_model_mean_temp.to_csv(path + '../tmp/train/global_model_mean_temp_' + self.control + '.csv', index=False)

    def compare_exception(self):
        """
        比较异常
        """
        global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp.csv')
        global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean.csv').rename(columns={'median_price': 'lately_median_price'})

        global_model_mean_temp = global_model_mean_temp.merge(global_model_mean.loc[:, ['detail_slug', 'lately_median_price']], how='left', on=['detail_slug'])
        global_model_mean_temp['difference_rate'] = abs(global_model_mean_temp['median_price'] - global_model_mean_temp['lately_median_price']) / global_model_mean_temp['lately_median_price']
        global_model_mean_temp = global_model_mean_temp.loc[(global_model_mean_temp['difference_rate'] > 0.05), :]
        global_model_mean_temp.to_csv(path + '../tmp/train/compare_exception.csv', index=False)

    def combine_final_detail(self):
        """
        组合最终款型
        """
        at = pd.read_csv(path + '../tmp/train/global_model_mean_temp_自动.csv')
        mt = pd.read_csv(path + '../tmp/train/global_model_mean_temp_手动.csv')
        final = at.append(mt, sort=False).reset_index(drop=True)
        final = final.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        final.to_csv(path + '../tmp/train/global_model_mean_temp.csv', index=False)
        print('最终组合款型数:', len(final))

    def execute(self):
        """
        执行
        """
        # self.handle_data_quality()
        # self.handle_data_preprocess()
        ## self.update_price_bn_div_map()
        ## self.update_warehouse_years_div_map()

        # self.update_model_map()
        # self.update_retain_high_config()
        # self.ttp_reverse_examine()

        self.combine_final_detail()

