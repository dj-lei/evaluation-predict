from valuate.process import *


class Process(object):

    def process_all_models(self):
        """
        处理所有品牌
        """
        try:
            # 特征工程
            time1 = time.time()
            fe = FeatureEngineering()
            fe.execute()
            # 更新数据库
            # process_tables.update_all()
            time2 = time.time()
            print(time2-time1)

        except Exception as e:
            print(traceback.format_exc())

    def process_cron(self):
        """
        执行日常
        """
        try:
            # 特征工程
            time1 = time.time()
            fe = FeatureEngineeringCron()
            fe.execute()
            # 更新数据库
            # process_tables.update_all()
            time2 = time.time()
            print(time2-time1)

        except Exception as e:
            print(traceback.format_exc())