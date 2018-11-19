from valuate.predict import *


def transform_int(df):
    """
    转换成int
    """
    return int(df['price_bn'])


def transform_hedge(df):
    """
    拆分保值率供数据库使用
    """
    dealer_hedge = df['dealer_hedge']
    dealer_hedge = ast.literal_eval(dealer_hedge)
    cpersonal_hedge = df['cpersonal_hedge']
    cpersonal_hedge = ast.literal_eval(cpersonal_hedge)
    hedge = [[i, j] for i, j in zip(dealer_hedge, cpersonal_hedge)]
    return pd.Series(hedge)


def transform_data(data):
    """
    转换成数据库需要格式
    """
    mdm = model_detail_map.loc[:, ['price_bn', 'model_detail_slug', 'model_detail_slug_id', 'model_slug_id']]
    pcm = province_city_map.loc[:, ['city', 'city_id', 'province_id']]
    ppm = province_popularity_map.loc[:, ['popularity', 'model_slug_id', 'province_id']]

    data = data.merge(mdm, how='left', on=['model_detail_slug'])
    data = data.merge(pcm, how='left', on=['city'])
    data = data.merge(ppm, how='left', on=['model_slug_id', 'province_id'])
    data['popularity'] = data['popularity'].fillna('C')
    data['price_bn'] = data['price_bn'] * 10000
    # data['price_bn'] = data['price_bn'].astype(int)
    data['price_bn'] = data.apply(transform_int, axis=1)
    data = data.drop(['model_detail_slug', 'city'], axis=1)
    return data


class Predict(object):

    def __init__(self):
        """
        加载各类匹配表和模型
        """
        self.test_level1 = pd.DataFrame()
        self.test_level2 = []
        self.predict_hedge = []
        self.predict_price = []
        self.valuate_model = []

        self.province_city_map = pd.read_csv(path + 'predict/map/province_city_map.csv')
        self.province_city_map = self.province_city_map.drop('city_id', axis=1)

    def create_test_data(self, model_slug):
        """
        创建测试数据
        """
        # 加载编码
        encode_model_detail_slug = pd.read_csv(path+'predict/model/feature_encode/model_detail_slug.csv')

        test = pd.DataFrame()
        for model_detail_slug in list(set(encode_model_detail_slug.model_detail_slug.values)):
            temp = self.province_city_map.copy()
            temp['model_detail_slug'] = model_detail_slug
            test = test.append(temp)
        test['model_slug'] = model_slug
        # test['popularity'] = test.apply(find_popularity, axis=1)
        test['source_type'] = 'dealer'
        result = test.copy()
        test['source_type'] = 'cpersonal'
        result = result.append(test)
        result = result.drop(['model_slug', 'province'], axis=1)
        result.to_csv(path+'predict/model/data/test.csv', index=False, encoding='utf-8')

    def predict_test_data(self, model_slug):
        """
        预测测试数据
        """
        try:
            # 创建测试数据
            self.create_test_data(model_slug)

            # 加载预测数据
            test = pd.read_csv(path+'predict/model/data/test.csv')
            # 加载估值模型
            self.valuate_model = xgb.Booster()
            self.valuate_model.load_model(path+'predict/model/model/xgboost_level2.model')
            # 加载编码
            encode_city = pd.read_csv(path+'predict/model/feature_encode/city.csv')
            encode_model_detail_slug = pd.read_csv(path+'predict/model/feature_encode/model_detail_slug.csv')
            encode_source_type = pd.read_csv(path+'predict/model/feature_encode/source_type.csv')
            test = test.merge(encode_city, how='left', on='city')
            test = test.merge(encode_model_detail_slug, how='left', on='model_detail_slug')
            test = test.merge(encode_source_type, how='left', on='source_type')

            # 加载特征顺序
            with open(path+'predict/model/feature_encode/feature_order.txt', 'rb') as fp:
                feature_name = pickle.load(fp)

            # 组合预测数据
            used_years = pd.DataFrame({'used_years': range(1, 21)})
            test['key'] = 1
            used_years['key'] = 1
            test = pd.merge(test, used_years, on='key')

            # 预测保值率
            temp = test.loc[:, feature_name]
            self.predict_hedge = np.exp(self.valuate_model.predict(xgb.DMatrix(temp)))
            temp['predict_hedge'] = pd.Series(self.predict_hedge).values
            temp['predict_hedge'] = temp['predict_hedge'].map('{:,.3f}'.format)

            # 整合保值率
            values = list(temp.predict_hedge.values)
            hedge = [str(values[i:i + 20]).replace("'", "") for i in range(0, len(values), 20)]
            test = test.drop_duplicates(['city', 'model_detail_slug', 'source_type'])
            test.reset_index(inplace=True, drop=True)
            test['predict_hedge'] = pd.Series(hedge)

            # 组合b2c,c2c数据
            test = test.drop(['source_type_encode', 'model_detail_slug_encode', 'city_encode'], axis=1)
            test_dealer = test.loc[(test['source_type'] == 'dealer'), ['model_detail_slug', 'city', 'predict_hedge']]
            test_dealer = test_dealer.rename(columns={'predict_hedge': 'dealer_hedge'})

            test_cpersonal = test.loc[(test['source_type'] == 'cpersonal'), ['model_detail_slug', 'city', 'predict_hedge']]
            test_cpersonal = test_cpersonal.rename(columns={'predict_hedge': 'cpersonal_hedge'})

            result = test_dealer.merge(test_cpersonal, how='left', on=['model_detail_slug', 'city'])

            # 整合max_year
            model_train = pd.read_csv(path+'predict/model/data/train.csv')
            model_train = model_train.loc[:, ['max_year', 'model_detail_slug']]
            model_train = model_train.sort_values(by=['model_detail_slug', 'max_year'], ascending=False)
            model_train = model_train.drop_duplicates(['model_detail_slug'])
            result = result.merge(model_train, how='left', on='model_detail_slug')

            # 拆分保值率
            columns_names = ['used_years_' + str(i) for i in range(1, 21)]
            result[columns_names] = result.apply(transform_hedge, axis=1)
            result = result.drop(['dealer_hedge', 'cpersonal_hedge'], axis=1)
            result = transform_data(result)
            result.to_csv(path + 'predict/model/data/result.csv', index=False, encoding='utf-8', float_format='%.3f')
        except Exception:
            raise ModelSlugPredictModelError(model_slug, traceback.format_exc())






