from evaluation.db import *


def insert_or_update_brand_model():
    """
    插入或更新品牌车系
    """
    combine_brand = pd.read_csv(path + '../tmp/train/combine_brand.csv')
    combine_model = pd.read_csv(path + '../tmp/train/combine_model.csv')
    combine_brand = combine_brand.drop(['car_autohome_brand_id'], axis=1)
    combine_model = combine_model.drop(['car_autohome_model_id'], axis=1)

    base_standard_open_category = combine_brand.append(combine_model, sort=False).reset_index(drop=True)
    base_standard_open_category = base_standard_open_category.sort_values(by=['id']).reset_index(drop=True)
    db_operate.insert_or_update_base_standard_open_category(base_standard_open_category)


def insert_or_update_detail():
    """
    插入或更新款型库
    """
    combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv')

    open_model_detail = pd.read_csv(path + '../tmp/train/open_model_detail.csv')
    open_model_detail = open_model_detail.loc[~(open_model_detail['model_detail_slug'].isin(list(set(combine_detail.detail_model_slug.values)))), :]
    open_model_detail = open_model_detail.rename(columns={'model_detail_slug': 'detail_model_slug'})
    open_model_detail['status'] = 'D'
    combine_detail = combine_detail.append(open_model_detail, sort=False)

    combine_detail = combine_detail.sort_values(by=['id']).reset_index(drop=True)
    combine_detail = combine_detail.drop(['car_autohome_detail_id'], axis=1)
    db_operate.insert_or_update_base_standard_open_model_detail(combine_detail)


def insert_global_model_mean():
    """
    插入全国均价表
    """
    at = pd.read_csv(path + '../tmp/train/global_model_mean_temp_自动.csv')
    mt = pd.read_csv(path + '../tmp/train/global_model_mean_temp_手动.csv')
    final = at.append(mt, sort=False).reset_index(drop=True)
    final = final.sort_values(by=['brand_slug', 'model_slug', 'online_year', 'price_bn']).reset_index(drop=True)
    final.to_csv(path + '../tmp/train/global_model_mean.csv', index=False)
    final.to_csv(path + '../tmp/train/global_model_mean_' + datetime.datetime.now().strftime("%Y-%m-%d") + '.csv', index=False)

    final = final.drop(['listed_year', 'update_time', 'control'], axis=1)
    db_operate.insert_valuate_global_model_mean(final)


def insert_province_city():
    """
    插入城市省份差异表
    """
    div_province_k_param = pd.read_csv(path + '../tmp/train/div_province_k_param.csv')
    db_operate.insert_valuate_province_city(div_province_k_param)


def insert_car_deal_history():
    """
    插入历史交易数据
    """
    car_deal_history = pd.read_csv(path + '../tmp/train/car_deal_history.csv')
    db_operate.insert_base_car_deal_history(car_deal_history)


def update_all():
    """
    更新数据到数据库
    """
    # insert_or_update_brand_model()
    # insert_or_update_detail()
    insert_global_model_mean()
    # insert_province_city()
    # insert_car_deal_history()

