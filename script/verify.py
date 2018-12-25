import pandas as pd
import numpy as np
import datetime
import os
os.environ['VALUATE_RUNTIME_ENVIRONMENT'] = 'TEST'

from valuate.predict.predict_api import Predict

if __name__ == "__main__":
    predict = Predict()
    verify = pd.read_csv('./verify.csv')
    verify['predict'] = np.NaN
    verify['predict_with_condition'] = np.NaN
    verify['history_price_trend'] = np.NaN
    verify['future_price_trend'] = np.NaN
    verify['residuals'] = np.NaN
    for i in range(0, len(verify)):
        city = verify.loc[i, 'city']
        model_detail_slug = verify.loc[i, 'detail_slug']
        reg_year = int(verify.loc[i, 'year'])
        reg_month = int(verify.loc[i, 'month'])
        deal_year = datetime.datetime.now().year
        deal_month = datetime.datetime.now().month
        mile = verify.loc[i, 'mile']

        print(i, model_detail_slug, city, reg_year, deal_year, mile)
        result = predict.predict(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year, reg_month=reg_month,deal_year=deal_year, deal_month=deal_month, mile=mile)
        verify.loc[i, 'predict'] = str(result)
        result = predict.predict_with_condition(condition_desc=gl.CONDITION_JSON, city=city,model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month,mile=mile)
        verify.loc[i, 'predict_with_condition'] = str(result)
        result = predict.history_price_trend(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        verify.loc[i, 'history_price_trend'] = str(result)
        result = predict.future_price_trend(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        verify.loc[i, 'future_price_trend'] = str(result)
        result = predict.residuals(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        verify.loc[i, 'residuals'] = str(result)

    verify.to_csv('./predict_result.csv', index=False)