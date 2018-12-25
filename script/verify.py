import pandas as pd
import numpy as np
import datetime
import os
os.environ['VALUATE_RUNTIME_ENVIRONMENT'] = 'LOCAL'

from valuate.predict.predict_api import Predict

if __name__ == "__main__":
    verify = pd.read_csv('./verify.csv')
    verify['predict'] = np.NaN
    verify['predict_with_condition'] = np.NaN
    verify['history_price_trend'] = np.NaN
    verify['future_price_trend'] = np.NaN
    verify['residuals'] = np.NaN

    content = input("是否重新开始验证(y/n):")
    if content == 'y':
        data = pd.DataFrame([], columns=verify.columns)
        data.to_csv('./predict_result.csv', index=False)
    else:
        predict_result = pd.read_csv('./predict_result.csv')
        verify = verify.loc[~(verify['detail_model_slug'].isin(list(predict_result.detail_model_slug.values))), :].reset_index(drop=True)

    predict = Predict()
    for i in range(0, len(verify)):
        data = pd.DataFrame([], columns=verify.columns)
        data.loc[0, :] = verify.loc[i, :].copy()
        city = verify.loc[i, 'city']
        model_detail_slug = verify.loc[i, 'detail_model_slug']
        reg_year = int(verify.loc[i, 'year'])
        reg_month = int(verify.loc[i, 'month'])
        deal_year = datetime.datetime.now().year
        deal_month = datetime.datetime.now().month
        mile = int(verify.loc[i, 'mile'])

        print(i, model_detail_slug, city, reg_year, deal_year, mile)
        result = predict.predict(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year, reg_month=reg_month,deal_year=deal_year, deal_month=deal_month, mile=mile)
        data['predict'] = str(result)
        result = predict.predict_with_condition(condition_desc='[{"item":"左前纵梁","number":0},{"item":"右前纵梁","number":0},{"item":"左前减震器座","number":0},{"item":"右前减震器座","number":0},{"item":"防火墙","number":0},{"item":"左A柱","number":0},{"item":"左B柱","number":0},{"item":"左C柱","number":0},{"item":"右B柱","number":0},{"item":"右C柱","number":0},{"item":"右A柱","number":0},{"item":"左后纵梁","number":0},{"item":"右后纵梁","number":0},{"item":"右后减震器座","number":0},{"item":"左后减震器座","number":0},{"item":"火烧情况","number":0},{"item":"泡水情况","number":0},{"item":"前保险杠","number":0},{"item":"机盖","number":0},{"item":"左前叶子板","number":0},{"item":"左前轮胎","number":0},{"item":"左前轮毂","number":0},{"item":"前挡风玻璃","number":0},{"item":"左前门","number":0},{"item":"左后门","number":0},{"item":"左后叶子板","number":0},{"item":"左后轮胎","number":0},{"item":"左后轮毂","number":0},{"item":"左后大灯","number":0},{"item":"右后大灯","number":0},{"item":"后保险杠","number":0},{"item":"右后轮胎","number":0},{"item":"右后轮毂","number":0},{"item":"右后叶子板","number":0},{"item":"右后门","number":0},{"item":"右前门","number":0},{"item":"右前叶子板","number":0},{"item":"右前轮胎","number":0},{"item":"右前轮毂","number":0},{"item":"左侧下边梁","number":0},{"item":"右侧下边梁","number":0},{"item":"车顶","number":0},{"item":"右前大灯","number":0},{"item":"左前大灯","number":0},{"item":"中控台","number":0},{"item":"仪表盘","number":0},{"item":"主驾驶座椅","number":0},{"item":"副驾驶座椅","number":0},{"item":"后排座椅","number":0},{"item":"车辆顶棚","number":0},{"item":"换挡杆区域","number":0},{"item":"车辆娱乐设备","number":0},{"item":"方向盘","number":0},{"item":"空调","number":0},{"item":"右前翼子板内衬","number":0},{"item":"左前翼子板内衬","number":0},{"item":"右后翼子板内衬","number":0},{"item":"左后翼子板内衬","number":0},{"item":"前防撞梁","number":0},{"item":"后防撞梁","number":0},{"item":"备胎槽","number":0},{"item":"水箱框架","number":0},{"item":"发动机","number":0},{"item":"发动机状况","number":0},{"item":"变速箱","number":0},{"item":"变速箱状况","number":0},{"item":"转向系统","number":0},{"item":"离合器系统","number":0},{"item":"刹车系统","number":0},{"item":"发动机是否烧机油","number":0}]', city=city,model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month,mile=mile)
        data['predict_with_condition'] = str(result)
        result = predict.history_price_trend(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        data['history_price_trend'] = str(result)
        result = predict.future_price_trend(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        data['future_price_trend'] = str(result)
        result = predict.residuals(city=city, model_detail_slug=model_detail_slug, reg_year=reg_year,reg_month=reg_month, deal_year=deal_year, deal_month=deal_month, mile=mile)
        data['residuals'] = str(result)
        data.to_csv('./predict_result.csv', header=False, mode='a', index=False)
