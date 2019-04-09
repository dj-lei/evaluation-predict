from evaluation.process import *


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
            process_tables.update_all()
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

    def test(self):
        """
        测试
        """
        predict_local = PredictLocal()
        result = predict_local.predict(city='苏州', model_detail_slug='m8991_xc', reg_year=2011, reg_month=3, mile=5, ret_type='normal')
        print(result)