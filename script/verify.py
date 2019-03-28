import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import datetime
import os
os.environ['VALUATE_RUNTIME_ENVIRONMENT'] = 'LOCAL'

from match.predict.predict_api import Predict
from evaluation.predict.predict_local import PredictLocal


def process_time(df):
    year_month = str(df['year']).split('.')
    return pd.Series([int(year_month[0]), int(year_month[1])])


if __name__ == "__main__":
    verify = pd.read_csv('./verify_valuate.csv')
    verify['title'] = verify['origin_brand_name'] +' '+ verify['origin_model_name'] +' '+ verify['origin_detail_name']
    verify[['year', 'month']] = verify.apply(process_time, axis=1)
    condition = {"优秀": 'excellent', "较好": 'good', "一般": 'fair'}
    verify['condition'] = verify.condition.map(condition)

    result = pd.DataFrame()
    predict = Predict()
    valuate = PredictLocal()
    for i in range(0, len(verify)):
        print(i, verify['title'][i], verify['year'][i], verify['month'][i])
        temp = predict.predict(is_evaluation=True, detail_name=verify['title'][i], cos_similar=0,
                                 price=verify['price'][i], intent='sell', condition=verify['condition'][i], city=verify['city'][i], reg_year=int(verify['year'][i]), reg_month=int(verify['month'][i]),
                                 deal_year=datetime.datetime.now().year, deal_month=datetime.datetime.now().month,
                                 mile=verify['mile'][i])['data']
        temp = pd.DataFrame(temp, index=[0])
        temp['id'] = verify['id'][i]
        temp = temp.loc[:, ['id', 'brand_name', 'model_name', 'detail_name', 'gpj_detail_slug', 'eval_price', 'online_year']]

        valuate_result = valuate.predict(city=verify['city'][i], model_detail_slug=temp['gpj_detail_slug'][0], reg_year=int(verify['year'][i]), reg_month=int(verify['month'][i]), deal_year=datetime.datetime.now().year, deal_month=datetime.datetime.now().month, mile=verify['mile'][i], ret_type='normal')
        new_price= valuate_result.loc[(valuate_result['intent'] == 'sell'), verify['condition'][i]].values[0]
        temp['new_sell_price'] = float('%.2f' % (new_price / 10000))

        valuate_result = valuate.predict(city=verify['city'][i], model_detail_slug=temp['gpj_detail_slug'][0], reg_year=int(verify['year'][i]), reg_month=int(verify['month'][i]), deal_year=datetime.datetime.now().year, deal_month=datetime.datetime.now().month, mile=verify['mile'][i], ret_type='normal')
        new_price= valuate_result.loc[(valuate_result['intent'] == 'buy'), verify['condition'][i]].values[0]
        temp['new_buy_price'] = float('%.2f' % (new_price / 10000))
        result = result.append(temp, sort=False)
    verify = verify.merge(result, how='left', on=['id'])
    verify.to_csv('./man.csv', index=False)
