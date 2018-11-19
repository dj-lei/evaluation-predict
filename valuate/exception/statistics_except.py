from valuate.exception import *


def classify_hedge(df):
    """
    区分非正常保值率
    """
    if df['used_years'] == 1:
        if 0.65 < df['hedge'] <= 0.965:
            return 0
        else:
            return 1
    elif df['used_years'] == 2:
        if 0.55 < df['hedge'] <= 0.89:
            return 0
        else:
            return 1
    elif df['used_years'] == 3:
        if 0.4 < df['hedge'] <= 0.80:
            return 0
        else:
            return 1
    elif df['used_years'] == 4:
        if 0.3 < df['hedge'] <= 0.70:
            return 0
        else:
            return 1
    elif df['used_years'] == 5:
        if 0.30 < df['hedge'] <= 0.62:
            return 0
        else:
            return 1
    elif df['used_years'] == 6:
        if 0.25 < df['hedge'] <= 0.55:
            return 0
        else:
            return 1
    elif df['used_years'] == 7:
        if 0.15 < df['hedge'] <= 0.49:
            return 0
        else:
            return 1
    elif df['used_years'] == 8:
        if 0.08 < df['hedge'] <= 0.40:
            return 0
        else:
            return 1
    elif df['used_years'] == 9:
        if 0.05 < df['hedge'] <= 0.32:
            return 0
        else:
            return 1
    elif df['used_years'] == 10:
        if 0.04 < df['hedge'] <= 0.25:
            return 0
        else:
            return 1
    elif df['used_years'] == 11:
        if 0.04 < df['hedge'] <= 0.22:
            return 0
        else:
            return 1
    elif df['used_years'] == 12:
        if 0.04 < df['hedge'] <= 0.2:
            return 0
        else:
            return 1
    elif df['used_years'] == 13:
        if 0.03 < df['hedge'] <= 0.18:
            return 0
        else:
            return 1
    elif df['used_years'] == 14:
        if 0.03 < df['hedge'] <= 0.115:
            return 0
        else:
            return 1
    elif df['used_years'] == 15:
        if 0.03 < df['hedge'] <= 0.10:
            return 0
        else:
            return 1
    elif df['used_years'] == 16:
        if 0.03 < df['hedge'] <= 0.09:
            return 0
        else:
            return 1
    elif df['used_years'] == 17:
        if 0.02 < df['hedge'] <= 0.08:
            return 0
        else:
            return 1
    elif df['used_years'] == 18:
        if 0.02 < df['hedge'] <= 0.07:
            return 0
        else:
            return 1
    elif df['used_years'] == 19:
        if 0.02 < df['hedge'] <= 0.065:
            return 0
        else:
            return 1
    elif df['used_years'] == 20:
        if 0.02 < df['hedge'] <= 0.06:
            return 0
        else:
            return 1


class StatisticsExcept(object):

    def __init__(self):
        self.train = []
        self.residuals_exception_file = path + 'predict/model/data/residuals_exception.csv'
        self.hedge_exception_file = path + 'predict/model/data/hedge_exception.csv'
        self.configure_exception_file = path + 'predict/model/data/configure_exception.csv'

    def per_year_residuals_exception(self, data):
        """
        每年残值率衰减异常,原则上顺序递减
        """
        self.train = data.copy()
        result = pd.DataFrame()
        # 统计省份残值衰减异常
        for detail in list(set(self.train.model_detail_slug.values)):
            detail_data = self.train.loc[(self.train['model_detail_slug'] == detail), :]
            detail_data = detail_data.groupby(['brand_slug', 'model_slug', 'model_detail_slug', 'province', 'used_years'])[
                'hedge'].median().reset_index()
            model_detail_slug_city = detail_data.loc[:, ['brand_slug', 'model_slug', 'model_detail_slug', 'province']].drop_duplicates(
                ['brand_slug', 'model_slug', 'model_detail_slug', 'province'])
            for bs, ms, mds, province in (model_detail_slug_city.loc[:, ['brand_slug', 'model_slug', 'model_detail_slug', 'province']].values):
                temp_data = detail_data.loc[
                            (detail_data['model_detail_slug'] == mds) & (detail_data['province'] == province), :]
                temp_sort = list(temp_data.hedge.values)
                temp_sort.sort(reverse=True)
                if list(temp_data.hedge.values) != temp_sort:
                    result = result.append(pd.DataFrame([[fs.STATISTICAL_EXCEPT_RESIDUALS_SEQUENCE_PROVINCE, bs, ms, mds, province, str(list(temp_data.used_years.values)), str(list(temp_data.hedge.values))]], columns=['exception_category', 'brand_slug', 'model_slug', 'model_detail_slug', 'province', 'used_years', 'hedge']))
                    result.reset_index(inplace=True, drop=True)
        result.to_csv(self.residuals_exception_file, mode='a', header=False, index=False, float_format='%.3f')

    def used_year_hedge_exception(self, data):
        """
        车龄保值率异常
        """
        self.train = data.copy()
        # 保值率超范围
        self.train = self.train.groupby(['brand_slug', 'model_slug', 'model_detail_slug', 'province', 'discount', 'used_years'])['hedge'].median().reset_index()
        self.train.reset_index(inplace=True, drop='index')
        self.train['hedge_exception'] = self.train.apply(classify_hedge, axis=1)
        self.train = self.train.loc[(self.train['hedge_exception'] == 1), :]
        self.train = self.train.loc[:, ['brand_slug', 'model_slug', 'model_detail_slug', 'province', 'discount', 'used_years', 'hedge']]
        self.train['exception_category'] = fs.STATISTICAL_EXCEPT_USED_YEAR_HEDGE_PROVINCE

        self.train = self.train.loc[:, ['exception_category', 'brand_slug', 'model_slug', 'model_detail_slug', 'province', 'discount', 'used_years', 'hedge']]
        # 取最大和最小各一个
        part1 = self.train.loc[self.train.groupby(['model_detail_slug', 'used_years']).hedge.idxmax(), :]
        part2 = self.train.loc[self.train.groupby(['model_detail_slug', 'used_years']).hedge.idxmin(), :]
        result = part1.append(part2)
        result = result.sort_values(['model_detail_slug', 'used_years'])
        # 保值率大于促销折扣
        part3 = self.train.loc[(self.train['hedge'] >= self.train['discount']), :]
        result = result.append(part3)
        # result.to_csv(self.hedge_exception_file, index=False, float_format='%.3f')
        result.to_csv(self.hedge_exception_file, mode='a', header=False, index=False, float_format='%.3f')

    def upper_lower_configure_exception(self, data):
        """
        高低配异常
        """
        self.train = data.copy()
        # 保值率超范围
        self.train = self.train.groupby(['brand_slug', 'model_slug', 'model_detail_slug', 'province', 'used_years'])['hedge'].median().reset_index()
        self.train = self.train.merge(model_detail_map.loc[:, ['model_detail_slug', 'year', 'price_bn', 'control', 'volume']], how='left', on=['model_detail_slug'])
        self.train['price'] = self.train['price_bn'] * self.train['hedge']
        self.train.reset_index(inplace=True, drop='index')
        model_slug_year = self.train.loc[:, ['brand_slug', 'model_slug', 'province', 'year', 'used_years', 'control', 'volume']].drop_duplicates(['brand_slug', 'model_slug', 'province', 'year', 'used_years', 'control', 'volume'])
        result = pd.DataFrame()
        for brand_slug,model_slug,province,year,used_years,control,volume in (model_slug_year.loc[:, ['brand_slug', 'model_slug', 'province', 'year', 'used_years', 'control', 'volume']].values):
            temp_data = self.train.loc[(self.train['model_slug'] == model_slug) & (self.train['province'] == province) & (self.train['year'] == year) & (self.train['used_years'] == used_years)& (self.train['control'] == control)& (self.train['volume'] == volume), :]
            temp_data = temp_data.sort_values(['model_slug', 'year', 'control', 'volume', 'used_years', 'province', 'price_bn'])
            temp_data.reset_index(inplace=True, drop=True)
            temp_sort = list(temp_data.price.values)
            temp_sort.sort()
            if list(temp_data.price.values) != temp_sort:
                result = result.append(pd.DataFrame([[fs.STATISTICAL_EXCEPT_UPPER_LOWER_CONFIGURE, brand_slug, model_slug, province, year, used_years, control, volume, str(list(temp_data.model_detail_slug.values)),str(list(temp_data.price.values))]],
                                                columns=['exception_category', 'brand_slug', 'model_slug', 'province',
                                                         'year', 'used_years', 'control', 'volume', 'details', 'price']))
                result.reset_index(inplace=True, drop=True)
        result.to_csv(self.configure_exception_file, mode='a', header=False, index=False, float_format='%.3f')

    def hedge__exception(self, data):
        """
        保值率突变异常
        """
        pass