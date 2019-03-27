from evaluation.train import *


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
        return df['median_price']
    else:
        return price[0]


class FeatureEngineeringCron(object):

    def __init__(self):
        self.train = pd.read_csv(path + '../tmp/train/train.csv')
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
        self.train.to_csv(path + '../tmp/train/train_temp.csv', index=False)

    def update_model_map(self):
        """
        生成全款型全国均值模型
        """
        self.train = pd.read_csv(path + '../tmp/train/train_temp.csv')
        global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean.csv')
        global_model_mean = global_model_mean.loc[global_model_mean.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
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

        # 取低配数据
        low_config_car = median_price.loc[median_price.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
        low_config_car = low_config_car.drop_duplicates(['model_slug', 'online_year'])

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
            part1 = part1.append(car_autohome_temp, sort=False).reset_index(drop=True)

        low_config_car = part1.loc[part1.groupby(['brand_slug', 'model_slug', 'online_year']).price_bn.idxmin(), :].reset_index(drop=True)
        low_config_car = low_config_car.drop_duplicates(['model_slug', 'online_year']).reset_index(drop=True)
        global_model_mean['median_price'] = global_model_mean.apply(update_price, args=(low_config_car,), axis=1)

        # 调整指导价差,确保同条件下高配比低配价格高
        result = pd.DataFrame()
        for model_slug, online_year in global_model_mean.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = car_autohome_all.loc[(car_autohome_all['model_slug'] == model_slug) & (car_autohome_all['online_year'] == online_year), :].reset_index(drop=True)
            car_autohome_temp = car_autohome_temp.merge(global_model_mean.loc[:, ['detail_slug', 'median_price']],how='left', on=['detail_slug'])
            low_config_price, price_bn = car_autohome_temp.loc[(car_autohome_temp['median_price'].notnull()), ['median_price', 'price_bn']].values[0]
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp)):
                car_autohome_temp.loc[i, 'median_price'] = float('%.2f' % ((car_autohome_temp.loc[i, 'price_bn'] - price_bn) * k + low_config_price))
            result = result.append(car_autohome_temp, sort=False).reset_index(drop=True)
        result = result.drop_duplicates(['detail_slug'])
        result.to_csv(path + '../tmp/train/global_model_mean_temp.csv', index=False)

    def update_retain_high_config(self):
        """
        保留高配价格
        """
        def retain_high_config(df):
            normal_update_price = median_price.loc[
                (median_price['detail_slug'] == df['detail_slug']), 'median_price'].values
            if len(normal_update_price) == 0:
                return df['median_price']

            all_update_price = np.median(origin_train.loc[(origin_train['detail_slug'] == df['detail_slug']), 'price'])
            if normal_update_price[0] >= all_update_price:
                update_price = all_update_price
            else:
                update_price = normal_update_price[0]
            return update_price

        combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv', low_memory=False)
        origin_train = pd.read_csv(path + '../tmp/train/train_temp.csv')
        global_model_mean_temp = pd.read_csv(path + '../tmp/train/global_model_mean_temp.csv')
        div_price_bn_k_param = pd.read_csv(path + '../tmp/train/div_price_bn_k_param.csv')

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
        result_retain = pd.DataFrame()
        for model_slug, online_year in model_online.loc[:, ['model_slug', 'online_year']].values:
            car_autohome_temp = retain_high_config_price.loc[(retain_high_config_price['model_slug'] == model_slug) & (retain_high_config_price['online_year'] == online_year), :].reset_index(drop=True)
            used_years = car_autohome_temp.loc[0, 'used_years']
            k = div_price_bn_k_param.loc[(div_price_bn_k_param['used_years'] == used_years), ['k']].values[0]
            for i in range(0, len(car_autohome_temp) - 1):
                if car_autohome_temp.loc[i, 'median_price'] > car_autohome_temp.loc[i + 1, 'median_price']:
                    div_price_bn = car_autohome_temp.loc[i + 1, 'price_bn'] - car_autohome_temp.loc[i, 'price_bn']
                    car_autohome_temp.loc[i + 1, 'median_price'] = float('%.3f' % (div_price_bn * k + car_autohome_temp.loc[i, 'median_price']))
            result_retain = result_retain.append(car_autohome_temp, sort=False).reset_index(drop=True)

        final = result_retain.copy()
        final = final.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
        final = final.merge(combine_detail.loc[:, ['detail_model_slug', 'car_autohome_detail_id']].rename(columns={'car_autohome_detail_id': 'detail_slug'}), how='left', on=['detail_slug'])
        final.to_csv(path + '../tmp/train/global_model_mean_temp.csv', index=False)
        print('最终组合款型数:', len(final))

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

    def execute(self):
        """
        执行
        """
        self.handle_data_quality()
        self.handle_data_preprocess()
        self.update_model_map()
        self.update_retain_high_config()
        self.compare_exception()

