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
            # 特征工程
            # time1 = time.time()
            # fe = FeatureEngineering()
            # fe.execute()
            # time2 = time.time()

            time1 = time.time()
            process_tables.update_all()
            time2 = time.time()
            print(time2-time1)

        except Exception as e:
            print(traceback.format_exc())