from valuate.db import *


def insert_or_update_brand_model():
    """
    插入或更新品牌车系
    """
    combine_brand = pd.read_csv(path + '../tmp/train/combine_brand.csv')
    combine_model = pd.read_csv(path + '../tmp/train/combine_model.csv')

    base_standard_open_category = combine_brand.append(combine_model, sort=False).reset_index(drop=True)
    base_standard_open_category = base_standard_open_category.sort_values(by=['id']).reset_index(drop=True)
    base_standard_open_category = base_standard_open_category.drop(['car_autohome_brand_id', 'car_autohome_model_id'], axis=1)
    db_operate.insert_or_update_base_standard_open_category(base_standard_open_category)


def insert_or_update_detail():
    """
    插入或更新款型库
    """
    combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv')
    combine_detail = combine_detail.sort_values(by=['id']).reset_index(drop=True)
    combine_detail = combine_detail.drop(['car_autohome_detail_id'], axis=1)
    db_operate.insert_or_update_base_standard_open_model_detail(combine_detail)


def insert_global_model_mean():
    """
    插入全国均价表
    """
    global_model_mean = pd.read_csv(path + '../tmp/train/global_model_mean.csv')
    db_operate.insert_valuate_global_model_mean(global_model_mean)


def insert_province_city():
    """
    插入城市省份差异表
    """
    div_province_k_param = pd.read_csv(path + '../tmp/train/div_province_k_param.csv')
    db_operate.insert_valuate_province_city(div_province_k_param)


def update_all():
    """
    更新数据到数据库
    """
    insert_or_update_brand_model()
    insert_or_update_detail()
    insert_global_model_mean()
    insert_province_city()

