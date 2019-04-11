from evaluation.predict import *


def get_profit_rate(intent, popularity, price):
    """
    获取畅销系数
    """
    # 按畅销程度分级,各交易方式相比于标价的固定比例
    profits = gl.PROFITS
    profit = profits[popularity]

    # sell_price = 0.9 * price - 5670
    # rate = sell_price / price

    rate = 0.34 * math.e ** (-0.6 * math.log(price / 10000, math.e))
    if rate <= 0.101:
        rate = 0.101

    # 计算各交易方式的价格相比于标价的固定比例
    if intent == 'sell':
        # 商家收购价相比加权平均价的比例
        # profit_rate = 1 - profit[0] - profit[1]
        profit_rate = (1 - profit[0]) * (1-rate)
    elif intent == 'buy':
        # 商家真实售价相比加权平均价的比例
        profit_rate = 1 - profit[0]
    elif intent == 'release':
        # 建议标价相比加权平均价的比例
        profit_rate = 1
    elif intent == 'private':
        # C2C价格相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[2]
    elif intent == 'lowest':
        # 最低成交价相比加权平均价的比例
        # profit_rate = 1 - profit[0] - profit[1] - profit[3]
        profit_rate = (1 - profit[0]) * (1 - rate) - profit[3]
    elif intent == 'cpo':
        # 认证二手车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[8]
    elif intent == 'replace':
        # 4S店置换价相比加权平均价的比例
        profit_rate = 1 - profit[0] - profit[4]
    elif intent == 'auction':
        # 拍卖价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[5]
    elif intent == 'avg-buy':
        # 平均买车价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[7]
    elif intent == 'avg-sell':
        # 平均卖车价价相比加权平均价的差异比例
        profit_rate = 1 - profit[0] - profit[6]
    return profit_rate


def cal_intent_condition(prices, condition):
    """
    计算所有交易方式的4个级别车况价
    """
    if condition == 'excellent':
        df2 = pd.DataFrame([[1, 0.961, 0.913, 0.855]])
    elif condition == 'good':
        df2 = pd.DataFrame([gl.CAR_CONDITION_COEFFICIENT_VALUES])
    elif condition == 'fair':
        df2 = pd.DataFrame([[1.094, 1.052, 1, 0.936]])
    else:
        df2 = pd.DataFrame([[1.168, 1.123, 1.067, 1]])
    df1 = pd.DataFrame(prices)
    all_map = df1.dot(df2)
    all_map.columns = ['excellent', 'good', 'fair', 'bad']
    all_map['intent'] = pd.Series(gl.INTENT_TYPE).values
    all_map = all_map.loc[:, ['intent', 'excellent', 'good', 'fair', 'bad']]
    all_map[['excellent', 'good', 'fair', 'bad']] = all_map[['excellent', 'good', 'fair', 'bad']].astype(int)
    all_map['condition'] = condition
    return all_map


def process_profit_rate(df, column_name, price):
    """
    畅销系数处理
    """
    return get_profit_rate(df[column_name], df['popularity'], price)


def check_params_value(reg_year, reg_month, deal_year, deal_month, mile):
    """
    校验参数
    """
    # 校验mile
    if not ((isinstance(mile, int)) | (isinstance(mile, float))):
        raise ApiParamsTypeError('mile', mile, 'Mile must be int or float!')
    elif mile < 0:
        raise ApiParamsValueError('mile', mile, 'Mile must be greater than zero!')
    # 校验month
    if (not isinstance(reg_month, int)) or (not (1 <= reg_month <= 12)):
        raise ApiParamsTypeError('reg_month', reg_month, 'reg_month must be int and in [1,12]!')
    if (not isinstance(deal_month, int)) or (not (1 <= deal_month <= 12)):
        raise ApiParamsTypeError('deal_month', deal_month, 'deal_month must be int and in [1,12]!')
    # 校验year
    if not isinstance(reg_year, int):
        raise ApiParamsTypeError('reg_year', reg_year, 'reg_year must be int!')
    if not isinstance(deal_year, int):
        raise ApiParamsTypeError('deal_year', deal_year, 'deal_year must be int!')
    if (deal_year - reg_year) > 20:
        raise ApiParamsValueError('deal_year - reg_year', deal_year - reg_year, 'The years of Forecast must be in 20 years!')
    # 校验时间差至少使用1个月
    if (deal_year < reg_year) | ((deal_year == reg_year) & (deal_month < reg_month)):
        raise ApiParamsTypeError('deal_year,deal_month and reg_year,reg_month', 0, 'Use at least 1 month!')


class PredictLocal(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean_temp.csv')
        self.div_province_k_param = pd.read_csv(path + '../tmp/train/div_province_k_param.csv')
        self.province_city_map = pd.read_csv(path + '../tmp/train/province_city_map.csv')
        self.province_city_map = self.province_city_map.loc[:, ['province', 'city']]
        self.result = []

    def add_process_intent(self, final_price, used_years):
        """
        根据交易方式修正预测值
        """
        # 组合结果
        self.result = result_map.copy()
        self.result.loc[(self.result['intent'] == 'buy'), 'predict_price'] = final_price
        self.result['predict_price'] = self.result['predict_price'].fillna(final_price)

        self.result['popularity'] = 'A'
        self.result['profit_rate'] = self.result.apply(process_profit_rate, args=('intent', final_price), axis=1)
        self.result['buy_profit_rate'] = self.result.apply(process_profit_rate, args=('intent_source', final_price), axis=1)
        self.result['predict_price'] = self.result['predict_price'] / self.result['buy_profit_rate']
        self.result['predict_price'] = self.result['profit_rate'] * self.result['predict_price']

        # 车况判断两年以内优秀,8-3年良好,9-11年一般,12年以上较差
        if used_years <= 2:
            condition = 'excellent'
        elif 2 < used_years <= 8:
            condition = 'good'
        elif 8 < used_years <= 11:
            condition = 'fair'
        elif 11 < used_years:
            condition = 'bad'
        # 计算所有交易类型
        self.result = cal_intent_condition(self.result.predict_price.values, condition)

    def query(self, city='深圳', model_detail_slug='model_25023_cs', reg_year=2015, reg_month=3, deal_year=datetime.datetime.now().year, deal_month=datetime.datetime.now().month, mile=2):
        """
        查询
        """
        # 校验参数
        check_params_value(reg_year, reg_month, deal_year, deal_month, mile)

        # 查询对应条件预测
        self.result = self.global_model_mean.loc[(self.global_model_mean['detail_model_slug'] == model_detail_slug), ['online_year', 'median_price', 'control']].reset_index(drop=True)
        if len(self.result) == 0:
            raise ApiParamsValueError('model_detail_slug or city', 0, 'Unknown model or city!')
        online_year, median_price, control = self.result.loc[0, :].values
        k, b = self.div_province_k_param.loc[(self.div_province_k_param['city'] == city), ['k', 'b']].values[0]
        median_price = int(median_price * 10000)

        # 省份差异
        province_price = k * median_price + b

        # 注册年份差异
        if online_year >= datetime.datetime.now().year:
            warehouse_year = 0
        else:
            warehouse_year = reg_year - online_year

        if control == '自动':
            k = 0.055
        else:
            k = 0.08
        warehouse_price = (k * warehouse_year) * median_price

        # 公里数差异
        used_months = ((deal_year - reg_year) * 12 + deal_month - reg_month)
        used_years = deal_year - reg_year
        if used_months <= 6:
            used_months = 6
        k, b = -0.1931, 0.0263
        mile_price = (k * (mile / used_months) + b) * median_price

        final_price = median_price + province_price + warehouse_price + mile_price
        # print(k)
        # print('median_price', int(median_price))
        # print('province_price', int(province_price))
        # print('warehouse_price', int(warehouse_price))
        # print('mile_price', int(mile_price))
        # print('final_price', int(final_price))

        # 根据交易方式修正预测值
        self.add_process_intent(final_price, used_years)

    def predict(self, city='深圳', model_detail_slug='model_25023_cs', reg_year=2015, reg_month=3, deal_year=datetime.datetime.now().year, deal_month=datetime.datetime.now().month, mile=2, ret_type=gl.RETURN_RECORDS):
        """
        预测返回
        """
        self.query(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year, reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)

        if ret_type == gl.RETURN_RECORDS:
            return self.result.to_dict(gl.RETURN_RECORDS)
        else:
            return self.result