from valuate.process import *


class Process(object):

    def __init__(self):
        self.source = []
        self.result = []

    def process_all_models(self):
        """
        处理所有品牌
        """
        try:
            # 存储训练相关表
            process_tables.store_train_relative_data()
            # 生成训练相关表
            manual = Manual()
            manual.execute()

            # 特征工程
            # fe = FeatureEngineering()
            # time1 = time.time()

            # data = pd.read_csv(path + '../tmp/train/car_source.csv')
            # data = data.loc[(data['brand_slug'].notnull()), :]
            # data.reset_index(inplace=True, drop=True)
            # fe.execute_data_clean(data)
            #
            # fe.execute_common_train()

            # fe.execute_brand()
            #
            # time2 = time.time()
            # print(time2-time1)

            # 存储可预测车型并更新数据库
            # process_tables.store_models_divinable()
        except Exception as e:
            print(traceback.format_exc())