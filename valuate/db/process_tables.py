from valuate.db import *


def update_relative_data():
    """
    更新数据到数据库
    """
    combine_brand = pd.read_csv(path + '../tmp/train/combine_brand.csv')
    combine_model = pd.read_csv(path + '../tmp/train/combine_model.csv')
    combine_detail = pd.read_csv(path + '../tmp/train/combine_detail.csv')

    # 插入或更新品牌车系库
    # base_standard_open_category = combine_brand.append(combine_model, sort=False).reset_index(drop=True)
    # base_standard_open_category = base_standard_open_category.sort_values(by=['id']).reset_index(drop=True)
    # db_operate.insert_or_update_base_standard_open_category(base_standard_open_category)

    # 插入或更新款型库
    db_operate.insert_or_update_base_standard_open_model_detail(combine_detail)

    # base_standard_open_category['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



