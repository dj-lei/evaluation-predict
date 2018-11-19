from valuate.train import *


def calculate_used_years(df):
    """
    生成使用的年份
    """
    online_year = df['year']
    if str(df['sold_time']) == 'nan':
        if len(str(df['expired_at'])) > 10:
            transaction_time = datetime.datetime.strptime(str(df['expired_at']), "%Y-%m-%d %H:%M:%S")
        else:
            transaction_time = datetime.datetime.strptime(str(df['expired_at']), "%Y-%m-%d")
    else:
        if len(str(df['sold_time'])) > 10:
            transaction_time = datetime.datetime.strptime(str(df['sold_time']), "%Y-%m-%d %H:%M:%S")
        else:
            transaction_time = datetime.datetime.strptime(str(df['sold_time']), "%Y-%m-%d")
    used_years = transaction_time.year - online_year
    if used_years == 0:
        return 1
    else:
        return used_years


def calculate_hedge_rate(df):
    """
    计算保值率
    """
    return float('%.3f' % (df['price'] / df['price_bn']))


def calculate_mile_per_month(df):
    """
    计算每月公里数
    """
    return float('%.2f' % (df['mile'] / df['used_years']))


def calculate_promotion_price(df):
    """
    计算促销价
    """
    if str(df['promotion_price']) == 'nan':
        return float('%.2f' % (promotion_details[df['model_detail_slug']]))
    else:
        return float('%.2f' % (df['promotion_price']))


def calculate_discount(df):
    """
    计算折扣
    """
    return float('%.2f' % (df['promotion_price'] / df['price_bn']))


def calculate_box_q1_q3_min_max(list_data, value_name):
    """
    计算箱形图的分位值
    """
    list_data.sort()
    count = len(list_data)
    min_value = min(list_data)
    max_value = max(list_data)
    median_value = median(list_data)
    q1_pos = list_data[int(count*0.25)]
    q3_pos = list_data[int(count*0.75)]
    iqr = q3_pos - q1_pos
    upper_limit_abnormity = q3_pos + 1.5*iqr
    lower_limit_abnormity = q1_pos - 1.5*iqr
    dict_values = {'min': min_value, 'max': max_value, 'median': median_value, 'q1': q1_pos,
                   'q3': q3_pos, 'iqr': iqr, 'upper_limit_abnormity': upper_limit_abnormity, 'lower_limit_abnormity': lower_limit_abnormity}
    return dict_values[value_name]


def match_number_price_bn(df):
    """
    新车指导价价格分段编码
    """
    if df['price_bn'] <= 5:
        return fs.PRICE['0-5']
    elif 5 < df['price_bn'] <= 10:
        return fs.PRICE['5-10']
    elif 10 < df['price_bn'] <= 20:
        return fs.PRICE['10-20']
    elif 20 < df['price_bn'] <= 30:
        return fs.PRICE['20-30']
    elif 30 < df['price_bn'] <= 50:
        return fs.PRICE['30-50']
    elif 50 < df['price_bn']:
        return fs.PRICE['50plus']


def match_number_control(df):
    """
    变速箱编码
    """
    return fs.CONTROL[df['control']]


def match_number_volume(df):
    """
    排量编码
    """
    if df['volume'] == 0:
        return fs.VOLUME['0']
    elif 0 < df['volume'] <= 1:
        return fs.VOLUME['0-1']
    elif 1 < df['volume'] <= 2:
        return fs.VOLUME['1-2']
    elif 2 < df['volume'] <= 3:
        return fs.VOLUME['2-3']
    elif 3 < df['volume'] <= 4:
        return fs.VOLUME['3-4']
    elif 4 < df['volume']:
        return fs.VOLUME['4plus']


def match_number_emission_standard(df):
    """
    排放标准编码
    """
    return fs.EMISSION_STANDARD[df['emission_standard']]


def match_number_brand_area(df):
    """
    国系编码
    """
    return fs.BRAND_AREA[df['brand_area']]


def match_number_attribute(df):
    """
    合资/国产/进口编码
    """
    return fs.ATTRIBUTE[df['attribute']]


def match_number_popularity(df):
    """
    流行度编码
    """
    return fs.POPULARITY[df['popularity']]


def match_features_to_statistical(data, flag=None):
    """
    编码统计模型特征
    """
    # 对所有特征进行编码
    omd = model_detail_map.loc[:, ['model_detail_slug', 'year', 'volume', 'control', 'emission_standard', 'price_bn', 'brand_area', 'attribute']]
    data = data.merge(omd, how='left', on=['model_detail_slug'])

    pcm = province_city_map.loc[:, ['province_id', 'province']]
    pcm = pcm.drop_duplicates(['province_id', 'province'])
    data = data.merge(pcm, how='left', on=['province']).rename(columns={'province_id': 'province_encode'})
    data['brand_area_encode'] = data.apply(match_number_brand_area, axis=1)
    data['attribute_encode'] = data.apply(match_number_attribute, axis=1)
    data['price_bn_encode'] = data.apply(match_number_price_bn, axis=1)
    data['volume_encode'] = data.apply(match_number_volume, axis=1)
    data['control_encode'] = data.apply(match_number_control, axis=1)
    data['emission_standard_encode'] = data.apply(match_number_emission_standard, axis=1)
    data['popularity_encode'] = data.apply(match_number_popularity, axis=1)
    if flag != None:
        # 临时添加
        data.to_csv(path + 'predict/model/data/train_all.csv', index=False)
        data = data.loc[(data['volume'] > 0), :]
        data.reset_index(inplace=True, drop=True)
    data = data.drop(['price_bn', 'volume', 'control', 'emission_standard', 'brand_area', 'attribute'], axis=1)
    return data


def match_features_to_stacking(data, brand_slug):
    """
    编码训练特征到stacking
    """
    # 对所有特征进行编码
    omd = model_detail_map.loc[(model_detail_map['brand_slug'] == brand_slug), ['model_slug', 'model_detail_slug', 'year', 'price_bn']]
    omd = omd.sort_values(['model_slug', 'year', 'price_bn'])
    omd.reset_index(inplace=True, drop=True)
    omd.reset_index(inplace=True)
    omd = omd.rename(columns={'index': 'model_detail_slug_encode'})

    data = data.merge(omd.loc[:, ['model_detail_slug', 'model_detail_slug_encode']], how='left', on=['model_detail_slug'])
    pcm = province_city_map.loc[:, ['province_id', 'province']]
    pcm = pcm.drop_duplicates(['province_id', 'province'])
    data = data.merge(pcm, how='left', on=['province']).rename(columns={'province_id': 'province_encode'})
    data = data.loc[:, ['model_detail_slug', 'discount', 'price_bn', 'popularity', 'model_detail_slug_encode', 'province_encode', 'used_years', 'hedge']]
    return data


class FeatureEngineering(object):

    def __init__(self):
        self.train = []
        self.no_data_details = []
        self.x_all = []
        self.y_all = []
        # 加载各类相关表
        self.domain_priority = pd.read_csv(path+'../tmp/train/domain_priority.csv')

    ###########################
    # 数据初步清洗
    ###########################
    def handle_data_quality(self, brand_slug):
        """
        处理数据质量
        """
        # 删掉遗漏值记录
        # 删掉不一致的记录

        # 保留可信度高平台和车商的记录和最近一年的数据,目前平台没有管控起来
        domains = self.domain_priority[self.domain_priority['type'] == 2]
        domains = list(set(domains.domain.values))
        part1 = self.train[(self.train['domain'].isin(domains))]
        # 款型数少于5的,放入赶集等平台的数据
        brand_models = list(set(model_detail_map.loc[(model_detail_map['brand_slug'] == brand_slug), 'model_detail_slug'].values))
        cur_models = part1.groupby(['model_detail_slug'])['brand_slug'].count().reset_index().rename(columns={'brand_slug':'num'})
        cur_models = list(cur_models.loc[(cur_models['num'] >= 5), 'model_detail_slug'].values)
        miss_models = list(set(brand_models) ^ set(cur_models))
        domains = self.domain_priority[self.domain_priority['type'] == 1]
        domains = list(set(domains.domain.values))
        part2 = self.train[(self.train['domain'].isin(domains))&(self.train['model_detail_slug'].isin(miss_models))]
        self.train = part1.append(part2)
        self.train.reset_index(inplace=True, drop=True)
        # 删掉price<0的记录
        self.train = self.train[self.train['price'] > 0]
        self.train = self.train[self.train['price_bn'] > 0]
        # 删掉不在20年车龄,月份1-12之外记录
        self.train = self.train[(self.train['year'] > (datetime.datetime.now().year-20))&(self.train['year'] <= datetime.datetime.now().year)]
        self.train = self.train[self.train['month'].isin(list(np.arange(1, 13)))]
        # 删除掉未知城市,未知款型的记录
        cities = list(set(province_city_map.city.values))
        models = list(set(model_detail_map.model_detail_slug.values))
        self.train = self.train[self.train['source_type'].isin(['odealer', 'dealer', 'cpersonal', 'personal'])]
        self.train = self.train[self.train['city'].isin(cities)]
        self.train = self.train[self.train['model_detail_slug'].isin(models)]

        # 瓜子和人人车平台车源需要加上0.35万的服务费
        self.train.loc[(self.train['domain'] == 'guazi.com'), 'price'] = self.train['price'] + 0.35
        self.train.loc[(self.train['domain'] == 'renrenche.com'), 'price'] = self.train['price'] + 0.35
        self.train.reset_index(inplace=True, drop='index')

    def handle_data_preprocess(self):
        """
        数据预处理
        """
        self.train['used_years'] = self.train.apply(calculate_used_years, axis=1)
        self.train['hedge'] = self.train.apply(calculate_hedge_rate, axis=1)
        self.train['mile_per_year'] = self.train.apply(calculate_mile_per_month, axis=1)
        self.train = self.train[self.train['mile_per_year'] <= 2.5]
        # 计算促销价
        discount = self.train.loc[(self.train['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))), :]
        discount.reset_index(inplace=True, drop=True)
        discount = discount.merge(promotion_4s_price.loc[:, ['model_detail_slug', 'province', 'promotion_price']],
                                  how='left', on=['model_detail_slug', 'province'])
        if len(discount) != 0:
            discount['promotion_price'] = discount.apply(calculate_promotion_price, axis=1)
        self.train = self.train.drop(self.train[self.train['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))].index)
        self.train['promotion_price'] = self.train['price_bn']
        self.train = self.train.append(discount)
        self.train.reset_index(inplace=True, drop='index')
        # 计算折扣
        self.train['discount'] = self.train.apply(calculate_discount, axis=1)
        self.train = self.train.drop(['year', 'month', 'status', 'mile_per_year', 'sold_time'], axis=1)
        # 匹配流行度
        self.train = self.train.drop(['popularity'], axis=1)
        self.train = self.train.merge(province_popularity_map.loc[:, ['model_slug', 'province', 'popularity']], how='left', on=['model_slug', 'province'])
        self.train['popularity'] = self.train['popularity'].fillna('C')

    def handle_statistical_analysis(self):
        """
        统计分析
        """
        # 取款型省份车龄最近5条记录
        self.train['pub_time'] = pd.to_datetime(self.train['pub_time'])
        result = pd.DataFrame()
        for i in range(0, 5):
            temp = self.train.loc[self.train.groupby(['model_detail_slug', 'province', 'used_years']).pub_time.idxmax(), :]
            self.train = self.train.drop(temp.index, axis=0)
            self.train.reset_index(inplace=True, drop=True)
            result = result.append(temp)
        self.train = result.copy()
        self.train.reset_index(inplace=True, drop=True)

        # 少于3条记录的过滤
        filter = self.train.groupby(['model_detail_slug', 'province', 'used_years'])['pub_time'].count().reset_index().rename(columns={'pub_time':'num'})
        filter = filter.loc[(filter['num'] >= 3), :]
        filter['match'] = 1
        filter.reset_index(inplace=True, drop=True)
        self.train = self.train.merge(filter, how='left', on=['model_detail_slug', 'province', 'used_years'])
        self.train = self.train.loc[(self.train['match'] == 1), :]
        self.train = self.train.drop(['match'], axis=1)
        self.train.reset_index(inplace=True, drop=True)
        # # 统计所有款型的省份极差分布
        # extre_div = []
        # for details in set(self.train.model_detail_slug.values):
        #     detail_data = self.train.loc[(self.train['model_detail_slug'] == details), :]
        #     for years in set(detail_data.used_years.values):
        #         data = detail_data.loc[(detail_data['used_years'] == years), :]
        #         province_data = data.groupby(['province'])['hedge'].count().reset_index().rename(
        #             columns={'hedge': 'num'})
        #         province_data = province_data.loc[(province_data['num'] >= 1), :]
        #         if len(province_data) == 0:
        #             continue
        #         data = data.loc[(data['province'].isin(list(set(province_data.province.values)))), :]
        #         province_data = data.groupby(['province'])['hedge'].median().reset_index()
        #         extre_div.append([details, years, (max(province_data.hedge.values) - min(province_data.hedge.values)) / median(data.hedge.values)])
        # extre_div_distributed = pd.DataFrame(extre_div, columns=['model_detail_slug', 'used_years', 'extre_div'])
        #
        # # 统计各年份的极差q3分位点
        # per_year_extre_div = pd.DataFrame()
        # for used_year in set(extre_div_distributed.used_years.values):
        #     year_extre_div = extre_div_distributed.loc[(extre_div_distributed['used_years'] == used_year), 'extre_div'].values
        #     per_year_extre_div = per_year_extre_div.append(pd.DataFrame([[used_year, calculate_box_q1_q3_min_max(year_extre_div, 'median')]], columns=['used_years', 'extre_div']))
        extre_div_distributed = self.train.loc[:, ['model_detail_slug', 'used_years']]
        extre_div_distributed = extre_div_distributed.drop_duplicates(['model_detail_slug', 'used_years'])
        extre_div_distributed = extre_div_distributed.sort_values(['model_detail_slug', 'used_years'])
        per_year_extre_div = pd.read_csv(path+'predict/model/data/extre_div.csv')

        # 根据极差清洗疑似异常
        result = pd.DataFrame()
        for details, used_year in (extre_div_distributed.loc[:,['model_detail_slug','used_years']].values):
            detail_data = self.train.loc[(self.train['model_detail_slug'] == details)&(self.train['used_years'] == used_year), :]
            if len(detail_data) < 3:
                continue
            median_hedge = median(detail_data.hedge.values)
            median_extre_div = per_year_extre_div.loc[(per_year_extre_div['used_years'] == used_year), 'extre_div'].values[0]
            upper_limit_hedge = median_hedge * median_extre_div / 2+median_hedge
            lower_limit_hedge = -median_hedge * median_extre_div / 2 + median_hedge
            result = result.append(detail_data.loc[(detail_data['hedge'] >= lower_limit_hedge) & (detail_data['hedge'] <= upper_limit_hedge), :])
        self.train = result.copy()

    ###########################
    # 数据异常处理
    # 1.查找异常
    # 2.人工分析异常
    # 3.删除异常
    ###########################
    def handle_statistical_except(self, data):
        """
        处理统计异常
        """
        se = StatisticsExcept()
        se.per_year_residuals_exception(data)
        se.used_year_hedge_exception(data)
        se.upper_lower_configure_exception(data)

    def handle_statistical_reasonable_adjust(self):
        """
        合理的统计调整
        """
        self.train = pd.read_csv(path + 'predict/model/data/train_common.csv')
        self.train.columns = ['pub_time', 'brand_slug', 'model_slug', 'model_detail_slug', 'mile', 'city', 'province',
                      'popularity', 'domain','price', 'price_bn', 'source_type', 'expired_at', 'used_years', 'hedge', 'promotion_price','discount']

        # 保值率异常删除(删除保值率大于折扣的数据)
        self.train = self.train.loc[(self.train['discount'] > self.train['hedge']), :]
        self.train.reset_index(inplace=True, drop=True)

        # 残值异常调整
        self.train = self.train.groupby(['brand_slug', 'model_slug', 'model_detail_slug', 'province', 'popularity', 'discount', 'used_years'])['hedge'].median().reset_index()
        self.train = self.train.sort_values(['model_detail_slug','province','used_years'],ascending=[True,True,True])
        self.train.reset_index(inplace=True, drop=True)
        data_temp = self.train.sort_values(['model_detail_slug','province','hedge'],ascending=[True,True,False])
        data_temp.reset_index(inplace=True, drop=True)
        self.train['hedge'] = pd.Series(list(data_temp.hedge.values))

        self.train.reset_index(inplace=True, drop=True)
        self.train.to_csv(path + 'predict/model/data/train.csv', index=False)

    ###########################
    # 通用模型训练
    ###########################
    def train_common_model(self):
        """
        训练通用统计模型
        """
        self.train = pd.read_csv(path + 'predict/model/data/train.csv')
        # 统计各款型各年份中间保值率
        train_data = self.train.loc[:, ['model_detail_slug', 'province', 'used_years', 'popularity', 'discount', 'hedge']]
        # 对object对象进行编码
        train_data = match_features_to_statistical(train_data, flag='common')
        # 训练统计模型
        self.x_all = train_data.loc[:, fs.STATISTICAL_MODEL_TRAIN_FEATURE]
        self.y_all = np.log(train_data[fs.STATISTICAL_MODEL_TARGET_FEATURE])
        model.xgb_train(self.x_all, self.y_all)

    def train_brand_model(self, brand_slug):
        """
        训练通用统计模型
        """
        self.train = pd.read_csv(path + 'predict/model/data/train_temp.csv')
        # 对object对象进行编码
        self.train = match_features_to_stacking(self.train, brand_slug)
        # 训练统计模型
        self.x_all = self.train.loc[:, fs.BRAND_MODEL_TRAIN_FEATURE]
        self.y_all = np.log(self.train[fs.STATISTICAL_MODEL_TARGET_FEATURE])
        model.xgb_brand_train(self.x_all, self.y_all)

    def generate_details_data(self, data, brand_slug):
        """
        增加款型数据
        1.冷门车
        2.新款车
        """
        self.train = data.copy()

        # 生成冷门,新款车数据
        all_details = model_detail_map.loc[(model_detail_map['brand_slug'] == brand_slug), 'model_detail_slug'].values
        self.no_data_details = [detail for detail in all_details if detail not in set(self.train.model_detail_slug.values)]
        self.no_data_details = pd.DataFrame(pd.Series(self.no_data_details), columns=['model_detail_slug'])
        self.no_data_details['match'] = 1
        provinces = pd.DataFrame(pd.Series(list(set(province_city_map.loc[:, 'province'].values))), columns=['province'])
        provinces['match'] = 1
        self.no_data_details = self.no_data_details.merge(provinces, how='left', on=['match'])
        self.no_data_details = self.no_data_details.merge(model_detail_map.loc[:, ['model_slug', 'model_detail_slug']], how='left', on=['model_detail_slug'])
        self.no_data_details = self.no_data_details.merge(province_popularity_map.loc[:, ['province','model_slug','popularity']], how='left', on=['province','model_slug'])
        self.no_data_details['popularity'] = self.no_data_details['popularity'].fillna('C')
        self.no_data_details = match_features_to_statistical(self.no_data_details)
        used_years = pd.DataFrame(pd.Series(list(range(1, 21))), columns=['used_years'])
        used_years['match'] = 1
        self.no_data_details = self.no_data_details.merge(used_years, how='left', on=['match'])
        self.no_data_details = self.no_data_details.drop(['match'], axis=1)

        # 计算促销价
        self.no_data_details = self.no_data_details.merge(model_detail_map.loc[:, ['price_bn', 'model_detail_slug']],
                                                          how='left', on=['model_detail_slug'])
        discount = self.no_data_details.loc[(self.no_data_details['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))), :]
        discount.reset_index(inplace=True, drop=True)
        discount = discount.merge(promotion_4s_price.loc[:, ['model_detail_slug', 'province', 'promotion_price']],
                                  how='left', on=['model_detail_slug', 'province'])
        if len(discount) != 0:
            discount['promotion_price'] = discount.apply(calculate_promotion_price, axis=1)
        self.no_data_details = self.no_data_details.drop(self.no_data_details[self.no_data_details['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))].index)
        self.no_data_details['promotion_price'] = self.no_data_details['price_bn']
        self.no_data_details = self.no_data_details.append(discount)
        self.no_data_details.reset_index(inplace=True, drop='index')
        # 计算折扣
        self.no_data_details['discount'] = self.no_data_details.apply(calculate_discount, axis=1)

        # 预测通用款型
        self.no_data_details['hedge'] = model.xgb_predict(self.no_data_details.loc[:, fs.STATISTICAL_MODEL_TRAIN_FEATURE])

        # 统计各款型各年份中间保值率
        self.train = self.train.loc[:, ['model_detail_slug', 'province', 'popularity', 'discount', 'used_years', 'hedge']]
        # 对object对象进行编码
        if len(self.train) != 0:
            self.train = match_features_to_statistical(self.train)
        self.train = self.train.append(self.no_data_details)
        self.train = self.train.loc[:, ['model_detail_slug', 'province', 'popularity', 'discount', 'used_years', 'hedge']]
        self.train.reset_index(inplace=True, drop=True)
        self.train.to_csv(path + 'predict/model/data/train_temp.csv', index=False)

    def fill_data(self, sub_competitor_data):
        """
        装填训练数据
        """
        # 装填人工行情价或竞品数据
        def split_car_condition(df):
            temp = df['origin'].split('_')
            return temp[0]

        def cal_competitor_hedge(df):
            temp = json.loads(df['valuate_price'])
            temp = pd.DataFrame.from_dict(temp, orient='index')
            temp.reset_index(level=0, inplace=True)
            temp = temp.rename(columns={'index': 'origin', 0: 'price'})
            temp['car_condition'] = temp.apply(split_car_condition, axis=1)
            if len(temp) != 21:
                return np.NAN
            elif not isinstance(temp.loc[0, 'price'], float):
                return np.NAN

            temp = temp.sort_values(['car_condition', 'price'], ascending=False)
            temp.reset_index(inplace=True, drop=True)
            if df['used_years'] <= 2:
                return float(
                    '%.3f' % (temp.loc[(temp['car_condition'] == 'excellent'), 'price'].values[1] / df['price_bn'] / 0.95))
            elif 2 < df['used_years'] <= 8:
                return float('%.3f' % (temp.loc[(temp['car_condition'] == 'good'), 'price'].values[1] / df['price_bn'] / 0.95))
            elif 8 < df['used_years']:
                return float(
                    '%.3f' % (temp.loc[(temp['car_condition'] == 'normal'), 'price'].values[1] / df['price_bn'] / 0.95))

        mdm = model_detail_map.loc[:, ['model_detail_slug', 'price_bn']]
        sub_competitor_data['used_years'] = datetime.datetime.now().year - sub_competitor_data['year']
        sub_competitor_data.loc[(sub_competitor_data['used_years'] <= 0), 'used_years'] = 1
        sub_competitor_data['discount'] = 1
        sub_competitor_data = sub_competitor_data.merge(mdm, how='left', on=['model_detail_slug'])
        if len(sub_competitor_data) != 0:
            sub_competitor_data['hedge'] = sub_competitor_data.apply(cal_competitor_hedge, axis=1)
        else:
            sub_competitor_data['hedge'] = np.NAN
        sub_competitor_data = sub_competitor_data.drop(['year', 'month', 'valuate_price', 'price_bn'], axis=1)
        sub_competitor_data = sub_competitor_data.loc[(sub_competitor_data['hedge'].notnull()), :]

        # 替换我方数据
        self.train = self.train.loc[~(self.train['model_detail_slug'].isin(list(set(sub_competitor_data.model_detail_slug.values)))), :]
        self.train = self.train.append(sub_competitor_data)

        # 特征编码
        self.train.to_csv(path+'predict/model/data/train_temp.csv', index=False)

    def predict_test_data(self, brand_slug):
        """
        预测测试数据
        """
        try:
            # 加载所有款型
            details = model_detail_map.loc[(model_detail_map['brand_slug'] == brand_slug), 'model_detail_slug'].values

            result = pd.DataFrame()
            for model_detail_slug in list(set(details)):
                # 组合城市
                temp = province_city_map.loc[:, ['province']]
                temp = temp.drop_duplicates(['province'])
                temp.reset_index(inplace=True, drop=True)
                temp['model_detail_slug'] = model_detail_slug
                result = result.append(temp)
                result.reset_index(inplace=True, drop=True)

            # 组合车龄
            used_years = pd.DataFrame({'used_years': range(1, 21)})
            result['key'] = 1
            used_years['key'] = 1
            result = pd.merge(result, used_years, on='key')

            # 计算促销价
            result = result.merge(model_detail_map.loc[:, ['price_bn', 'model_detail_slug', 'model_slug']],how='left', on=['model_detail_slug'])
            discount = result.loc[(result['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))), :]
            discount.reset_index(inplace=True, drop=True)
            discount = discount.merge(promotion_4s_price.loc[:, ['model_detail_slug', 'province', 'promotion_price']],
                                      how='left', on=['model_detail_slug', 'province'])
            if len(discount) != 0:
                discount['promotion_price'] = discount.apply(calculate_promotion_price, axis=1)
            result = result.drop(result[result['model_detail_slug'].isin(list(set(promotion_4s_price.model_detail_slug.values)))].index)
            result['promotion_price'] = result['price_bn']
            result = result.append(discount)
            result.reset_index(inplace=True, drop='index')
            # 计算折扣
            result['discount'] = result.apply(calculate_discount, axis=1)

            # 组合流行度
            result = result.merge(province_popularity_map.loc[:, ['province', 'model_slug', 'popularity']], how='left', on=['model_slug', 'province'])
            result['popularity'] = result['popularity'].fillna('C')
            # 编码匹配
            result = match_features_to_stacking(result, brand_slug)
            result.to_csv(path + 'predict/model/data/test.csv', index=False, encoding='utf-8')

            # 加载预测数据
            test = pd.read_csv(path + 'predict/model/data/test.csv')

            # 预测保值率
            temp = test.loc[:, fs.BRAND_MODEL_TRAIN_FEATURE]
            temp['predict_hedge'] = model.xgb_brand_predict(temp.loc[:, fs.BRAND_MODEL_TRAIN_FEATURE])
            temp['predict_hedge'] = temp['predict_hedge'].map('{:,.3f}'.format)

            # 整合保值率
            values = list(temp.predict_hedge.values)
            hedge = [str(values[i:i + 20]).replace("'", "") for i in range(0, len(values), 20)]
            test = test.drop_duplicates(['model_detail_slug_encode', 'province_encode'])
            test.reset_index(inplace=True, drop=True)
            test['predict_hedge'] = pd.Series(hedge)

            # 组合特征
            test = test.merge(model_detail_map.loc[:, ['model_detail_slug', 'brand_slug_id', 'model_slug_id', 'model_detail_slug_id']], how='left', on=['model_detail_slug'])
            test = test.rename(columns={'province_encode': 'province_id'})
            test = test.drop(['used_years','hedge','model_detail_slug_encode','model_detail_slug'], axis=1)
            test['update_time'] = datetime.datetime.now()
            test['price_bn'] = test['price_bn'] * 10000
            test['price_bn'] = test['price_bn'].astype(int)
            test.to_csv(path + 'predict/model/data/result.csv', index=False, encoding='utf-8', float_format='%.3f')
        except Exception:
            print(traceback.format_exc())

    ###########################
    # 执行分类
    ###########################
    def execute_data_clean(self, data):
        """
        数据清洗
        """
        # 清空异常信息
        result = pd.DataFrame()
        result.to_csv(path + 'predict/model/data/residuals_exception.csv', index=False)
        result.to_csv(path + 'predict/model/data/hedge_exception.csv', index=False)
        result.to_csv(path + 'predict/model/data/configure_exception.csv', index=False)
        result.to_csv(path + 'predict/model/data/train_common.csv', index=False)

        for i, brand in enumerate(list(set(data.brand_slug.values))):
        # for i, brand in enumerate(['mitsubishi']):
            print(i, 'start brand data clean:', brand)
            self.train = data.loc[(data['brand_slug'] == brand), :]
            self.train.reset_index(inplace=True, drop=True)
            self.handle_data_quality(brand)
            if len(self.train) < 12:
                print(i, 'no brand data clean:', brand)
                continue
            self.handle_data_preprocess()
            self.handle_statistical_analysis()
            if len(self.train) <= 0:
                print(i, 'no brand data clean:', brand)
                continue
            self.train = self.train.loc[:,['pub_time', 'brand_slug', 'model_slug', 'model_detail_slug', 'mile', 'city', 'province',
                      'popularity', 'domain','price', 'price_bn', 'source_type', 'expired_at', 'used_years', 'hedge', 'promotion_price','discount']]
            self.train.to_csv(path + 'predict/model/data/train_common.csv', mode='a', header=False, index=False, float_format='%.3f')
            self.handle_statistical_except(self.train)
            print(i, 'finish brand data clean:', brand)

    def execute_common_train(self):
        """
        通用模型训练
        """
        self.handle_statistical_reasonable_adjust()
        self.train_common_model()

    def execute_brand(self):
        """
        执行全部流程
        """
        train_data = pd.read_csv(path + 'predict/model/data/train.csv')
        competitor_data = pd.read_csv(path + '../tmp/train/competitor_data.csv')
        # for i, brand in enumerate(list(set(model_detail_map.brand_slug.values))):
        for i, brand in enumerate(['benz']):
            print(i, 'start brand handle:', brand)
            data = train_data.loc[(train_data['brand_slug'] == brand), :]
            sub_competitor_data = competitor_data.loc[(competitor_data['brand_slug'] == brand), :]
            data.reset_index(inplace=True, drop=True)
            sub_competitor_data.reset_index(inplace=True, drop=True)
            # 预测没数据款型
            self.generate_details_data(data, brand)
            # 装填人工反馈数据
            self.fill_data(sub_competitor_data)
            # 训练品牌模型
            self.train_brand_model(brand)
            # 提前预测
            self.predict_test_data(brand)
            # 将相关数据存入数据库
            process_tables.store_relative_data_to_database()
            print(i, 'finish brand handle:', brand)