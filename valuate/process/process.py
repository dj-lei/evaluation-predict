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
            # process_tables.store_train_relative_data()
            # 生成训练相关表
            # manual = Manual()
            # manual.execute()

            # 特征工程
            time1 = time.time()
            fe = FeatureEngineering()
            fe.execute()
            time2 = time.time()
            print(time2-time1)

        except Exception as e:
            print(traceback.format_exc())