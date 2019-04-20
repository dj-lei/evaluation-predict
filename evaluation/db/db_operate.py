from evaluation.db import *


def insert_or_update_base_standard_open_category(data):
    """
    插入或更新
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.base_standard_open_category'
        con.execute(sql)
    con.close()

    data.to_sql(name='base_standard_open_category', if_exists='append', con=engine, index=False)

    engine = create_engine(gl.TEST_PINGJIA_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE pingjia.open_category'
        con.execute(sql)
    con.close()
    data['url'] = data['url'].fillna('http://test')
    data.to_sql(name='open_category', if_exists='append', con=engine, index=False)


def insert_or_update_base_standard_open_model_detail(data):
    """
    插入或更新
    """
    # engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    #
    # with engine.begin() as con:
    #     sql = 'TRUNCATE TABLE china_used_car_estimate.base_standard_open_model_detail'
    #     con.execute(sql)
    # con.close()
    #
    # data.to_sql(name='base_standard_open_model_detail', if_exists='append', con=engine, index=False)

    engine = create_engine(gl.TEST_PINGJIA_PINGJIA_ENGINE, encoding=gl.ENCODING)

    # with engine.begin() as con:
    #     sql = 'TRUNCATE TABLE pingjia.open_model_detail'
    #     con.execute(sql)
    # con.close()

    data.to_sql(name='open_model_detail', if_exists='append', con=engine, index=False)


def insert_valuate_global_model_mean(data):
    """
    插入
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.valuate_global_model_mean'
        con.execute(sql)
    con.close()

    data.to_sql(name='valuate_global_model_mean', if_exists='append', con=engine, index=False)


def insert_valuate_province_city(data):
    """
    插入
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.valuate_province_city'
        con.execute(sql)
    con.close()

    data.to_sql(name='valuate_province_city', if_exists='append', con=engine, index=False)


def insert_base_car_deal_history(data):
    """
    插入
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    with engine.begin() as con:
        sql = 'TRUNCATE TABLE china_used_car_estimate.base_car_deal_history'
        con.execute(sql)
    con.close()

    for i in range(0, int(len(data)/50000)+1):
        temp = data.loc[i*50000:(i+1)*50000-1, :].reset_index(drop=True)
        temp.to_sql(name='base_car_deal_history', if_exists='append', con=engine, index=False)


# def insert_or_update_base_standard_open_category(data):
#     """
#     插入或更新
#     """
#     engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
#
#     columns_update = [column_name + '=' + 'VALUES(' + column_name + ')' for column_name in list(data.columns)]
#     columns_update = str(columns_update).replace('\'', '')
#     columns_update = columns_update[1:len(columns_update) - 1]
#
#     columns_name = str(list(data.columns)[1:])
#     columns_name = columns_name[1:len(columns_name) - 1]
#     columns_name = columns_name.replace('\'', '')
#     with engine.begin() as con:
#         for i in range(0, len(data)):
#             if str(data.loc[i, 'id']) == 'nan':
#                 value = str([v if str(v) != 'nan' else 'null' for v in list(data.loc[i, :].values)][1:])
#                 value = value[1:len(value) - 1]
#                 value = re.sub(r'\'null\'', 'null', value)
#                 sql = 'INSERT INTO china_used_car_estimate.base_standard_open_category (' + columns_name + ') VALUES (' + value + ')'
#             else:
#                 value = str(list(data.loc[i, :].values))
#                 value = value[1:len(value) - 1]
#                 value = re.sub(r' nan', ' null', value)
#                 sql = 'INSERT INTO china_used_car_estimate.base_standard_open_category VALUES (' + value + ') ON DUPLICATE KEY UPDATE ' + columns_update
#             con.execute(sql)
#     con.close()
#
#
# def insert_or_update_base_standard_open_model_detail(data):
#     """
#     插入或更新
#     """
#     engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
#
#     columns_update = [column_name + '=' + 'VALUES(' + column_name + ')' for column_name in list(data.columns)]
#     columns_update = str(columns_update).replace('\'', '')
#     columns_update = columns_update[1:len(columns_update) - 1]
#
#     columns_name = str(list(data.columns)[1:])
#     columns_name = columns_name[1:len(columns_name) - 1]
#     columns_name = columns_name.replace('\'', '')
#     with engine.begin() as con:
#         for i in range(0, len(data)):
#             if str(data.loc[i, 'id']) == 'nan':
#                 value = str([v if str(v) != 'nan' else 'null' for v in list(data.loc[i, :].values)][1:])
#                 value = value[1:len(value) - 1]
#                 value = re.sub(r'\'null\'', 'null', value)
#                 sql = 'INSERT INTO china_used_car_estimate.base_standard_open_model_detail (' + columns_name + ') VALUES (' + value + ')'
#             else:
#                 value = str(list(data.loc[i, :].values))
#                 value = value[1:len(value) - 1]
#                 value = re.sub(r' nan', ' null', value)
#                 sql = 'INSERT INTO china_used_car_estimate.base_standard_open_model_detail VALUES (' + value + ') ON DUPLICATE KEY UPDATE ' + columns_update
#             con.execute(sql)
#     con.close()