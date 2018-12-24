from valuate.db import *


###############################
# 生产库相关操作
###############################


def query_valuate(model_detail_slug, city, used_years):
    """
    查询生产库最近交易数据
    """
    used_years_label = 'used_years_'+str(used_years)

    query_sql = 'select vpd.'+used_years_label+',vpd.model_slug_id,vpd.province_id,vpd.city_id,vpd.price_bn,vpd.max_year,vpd.popularity,vccd.diff,vclsd.a,vclsd.b from valuate.valuate_predict_data as vpd \
        left join pingjia.open_model_detail as omd on vpd.model_detail_slug_id = omd.id \
        left join pingjia.open_city as oc on vpd.city_id = oc.id and oc.parent != 0 \
        left join valuate.valuate_c2b_calculate_data as vccd on vpd.model_slug_id = vccd.model_slug_id and vpd.province_id = vccd.province_id \
        left join valuate.valuate_c2b_least_squares_data as vclsd on vpd.popularity = vclsd.popularity \
        where omd.detail_model_slug = \''+model_detail_slug+'\' and oc.name = \''+city+'\' '

    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_valuate_all(model_detail_slug_id, city_id):
    """
    查询生产库最近交易数据
    """
    query_sql = 'select * from valuate_predict_data where model_detail_slug_id = '+str(model_detail_slug_id)+' and city_id = '+str(city_id)
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_recently_review_car_source(start_time, end_time):
    """
    查询生产库最近零售价和个人交易价数据
    """
    query_sql = 'select cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.dealer_id,cs.url,cs.expired_at,cs.sold_time from car_source as cs ' \
                'left join car_detail_info as cdi on cs.id = cdi.car_id ' \
                'where cdi.mdn_status = \'P\' and cs.global_sibling = 0 and cs.model_detail_slug is not null and status = \'review\' and sold_time >= \''+ start_time +'\' and sold_time <= \''+ end_time +'\' '
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_recently_review_deal_records(start_time, end_time):
    """
    查询生产库最近收购价的交易数据
    """
    query_sql = 'select id,model_detail_slug,mile,reg_date,city,price,status,source,deal_date from deal_records ' \
                    ' where deal_type = 2 and status = 1 and deal_date >= \''+ start_time +'\' and deal_date <= \''+ end_time +'\' '
    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)

###############################
# 测试库相关操作
###############################


def insert_valuate_model_detail_map(data):
    """
    存储可预测车型款型匹配表
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d')

    sql = 'delete from '+gl.VALUATE_MODEL_DETAIL_MAP+' where create_time <= \''+cur_time+'\' '
    with engine.begin() as con:
        con.execute(sql)
    con.close()
    time.sleep(0.2)
    data.to_sql(name=gl.VALUATE_MODEL_DETAIL_MAP, if_exists='append', con=engine, index=False)


def query_valuate_model_detail_map():
    """
    查询可预测车型款型匹配表
    """
    query_sql = 'select * from valuate_model_detail_map'
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def insert_valuate_predict_data(data, brand_slug_id):
    """
    存储预测数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    sql = 'delete from '+gl.VALUATE_PREDICT_DATA+' where brand_slug_id = '+str(brand_slug_id)
    with engine.begin() as con:
        con.execute(sql)
    con.close()
    time.sleep(0.2)
    data.to_sql(name=gl.VALUATE_PREDICT_DATA, if_exists='append', con=engine, index=False)


def insert_valuate_residuals_data(data, model_slug_id):
    """
    存储预测数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    sql = 'delete from '+gl.VALUATE_RESIDUALS_DATA+' where model_slug_id = '+str(model_slug_id)
    with engine.begin() as con:
        con.execute(sql)
    con.close()
    time.sleep(0.2)
    data.to_sql(name=gl.VALUATE_RESIDUALS_DATA, if_exists='append', con=engine, index=False)


def query_valuate_predict_data_divinable():
    """
    查询可预测款型
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    query_sql = 'select distinct model_detail_slug_id from valuate_predict_data_alter'
    return pd.read_sql_query(query_sql, engine)


def insert_valuate_error_info(e):
    """
    存储异常情况
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    error_type = e.error_type
    model_slug = e.model_slug
    message = e.message
    state = 'unprocessed'
    create_time = datetime.datetime.now()
    with engine.begin() as con:
        con.execute("""
           INSERT INTO valuate_error_history (error_type, model_slug, description, state, create_time) VALUES (%s, %s, %s, %s, %s)
        """, (error_type, model_slug, message, state, create_time))
    con.close()


def query_valuate_error_info():
    """
    存储差异数据到数据库
    """
    query_sql = 'select * from '+gl.VALUATE_ERROR_HISTORY+' where state = \'unprocessed\' '
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def update_valuate_error_info(record_id):
    """
    存储差异数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    state = 'processed'

    with engine.begin() as con:
        con.execute("""
           UPDATE valuate_error_history
           SET state=%s
           WHERE id=%s 
        """, (state, str(record_id)))
    con.close()


def insert_valuate_difference_data(data):
    """
    存储差异数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    data.to_sql(name=gl.VALUATE_DIFFERENCE_DATA, if_exists='append', con=engine, index=False)


def query_valuate_difference_data(process_status='A'):
    """
    存储差异数据到数据库
    """
    query_sql = 'select * from '+gl.VALUATE_DIFFERENCE_DATA+' where process_status = \''+process_status+'\''
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def update_valuate_difference_data(record_id, artificial_price):
    """
    存储差异数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    process_status = 'P'
    cur_time = datetime.datetime.now()
    expired_time = (datetime.datetime.now() + datetime.timedelta(days=180))

    with engine.begin() as con:
        con.execute("""
           UPDATE valuate_difference_data
           SET artificial_price=%s, process_status=%s, expired_time=%s, update_time=%s
           WHERE id=%s 
        """, (str(artificial_price), process_status, expired_time, cur_time, str(record_id)))
    con.close()


def update_valuate_difference_data_after_train():
    """
    存储差异数据到数据库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d')

    with engine.begin() as con:
        con.execute("""
           UPDATE valuate_difference_data
           SET process_status=\'Y\'
           WHERE process_status=\'P\' and create_time < %s
        """, (cur_time))
    con.close()


def insert_add_data(data):
    """
    新增人工新增车源
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    data['expired_time'] = (datetime.datetime.now() + datetime.timedelta(days=180))
    data['sold_time'] = datetime.datetime.now()
    data['mile'] = 0
    data['status'] = 'P'
    data.to_sql(name=gl.VALUATE_ADD_DATA, if_exists='append', con=engine, index=False)


def query_add_data():
    """
    查询人工新增车源
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    query_sql = 'select * from ' + gl.VALUATE_ADD_DATA + ' where status != \'N\' order by sold_time desc'
    return pd.read_sql_query(query_sql, engine)


def update_add_data():
    """
    更新人工新增车源
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d')

    with engine.begin() as con:
        con.execute("""
           UPDATE valuate_add_data
           SET status=\'Y\'
           WHERE status=\'P\' and sold_time < %s
        """, (cur_time))
    con.close()


def insert_operation_record(model_slug, operation_type, ):
    """
    新增人工新增车源
    """
    cur_time = datetime.datetime.now()
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    with engine.begin() as con:
        con.execute("""
           INSERT INTO valuate_operation_history (model_slug, type, create_time) VALUES (%s, %s, %s)
        """, (model_slug, operation_type, cur_time))
    con.close()


###############################
# 训练相关数据库操作
###############################


def query_all_history_table_name():
    """
    查询历史库训练数据
    """
    query_sql = 'select table_name from valuate.car_source_history_tables'
    engine = create_engine(gl.HISTORY_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_history_review_source_data(start_time, table, model_slug):
    """
    查询历史库训练数据
    """
    query_sql = 'select cs.title,cs.pub_time,cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.dealer_id,cs.expired_at,cs.sold_time ' \
                        'from '+ table +' as cs where cs.status = \'review\' and cs.global_sibling = 0 and cs.model_detail_slug is not null and model_slug = \''+model_slug+'\' and cs.sold_time >= \''+start_time+'\' '

    engine = create_engine(gl.HISTORY_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_product_review_source_data(start_time, model_slug):
    """
    查询生产库交易训练数据
    """
    query_sql = 'select cs.title,cs.pub_time,cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.dealer_id,cs.expired_at,cs.sold_time ' \
                        'from car_source as cs where cs.status = \'review\' and cs.global_sibling = 0 and cs.model_detail_slug is not null and model_slug = \''+model_slug+'\' and cs.sold_time >= \''+start_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_model_product_sale_source_data(start_time, model_slug):
    """
    查询生产库在售训练数据
    """
    query_sql = 'select cs.title,cs.pub_time,cs.id,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.price,cs.status,cs.source_type,cs.domain,cs.dealer_id,cs.expired_at,cs.sold_time ' \
                        'from car_source as cs where cs.status = \'sale\' and cs.global_sibling = 0 and cs.model_detail_slug is not null and model_slug = \''+model_slug+'\' and cs.pub_time >= \''+start_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_competitor_data():
    """
    查询竞品数据
    """
    # pub_time = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')

    query_sql = 'select vmcod.brand_slug,vmcod.model_detail_slug,vmcod.year,vmcod.month,vmcod.province,vmcod.popularity,vmcd.valuate_price from valuate_management_competitor_origin_data as vmcod ' \
                    'left join valuate_management_competitor_data as vmcd on vmcd.car_id = vmcod.id ' \
                    ' where vmcd.competitor_id = 2 and vmcd.valuate_price is not null '

    engine = create_engine(gl.PRODUCE_VALUATE_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_depreciation():
    """
    查询系数衰减表
    """
    query_sql = 'select * from open_depreciation'

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_province_popularity():
    """
    查询车型省份流行度表
    """
    query_sql = 'select * from open_province_popularity'

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_city():
    """
    查询省份城市表
    """
    query_sql = 'select id,name,parent from open_city'

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_4s_price_clear():
    """
    查询促销价
    """
    query_sql = 'select id,detail_model_slug,province,price,price_bn from open_4s_price_clear'

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_deal_records(start_time):
    """
    查询生产成交记录表
    """
    query_sql = 'select id,model_detail_slug,city,mile,price,deal_date,reg_date,source,status from pingjia.deal_records' \
                ' where deal_type = 2 and status = 1 and deal_date >= \''+start_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def insert_c2b_calculate_data(data):
    """
    存储可预测车型款型匹配表
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sql = 'delete from '+gl.VALUATE_C2B_CALCULATE_DATA+' where create_time < \''+cur_time+'\' '
    with engine.begin() as con:
        con.execute(sql)
    con.close()
    time.sleep(0.2)

    data.to_sql(name=gl.VALUATE_C2B_CALCULATE_DATA, if_exists='append', con=engine, index=False)


def insert_c2b_least_squares_data(data):
    """
    存储可预测车型款型匹配表
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    sql = 'delete from '+gl.VALUATE_C2B_LEAST_SQUARES_DATA+' where create_time < \''+cur_time+'\' '
    with engine.begin() as con:
        con.execute(sql)
    con.close()
    time.sleep(0.2)

    data.to_sql(name=gl.VALUATE_C2B_LEAST_SQUARES_DATA, if_exists='append', con=engine, index=False)


def query_produce_car_source():
    """
    查询款型库
    """
    pub_time = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')

    # query_sql = 'select cs.pub_time,cs.brand_slug,cs.model_slug,cs.model_detail_slug,cs.mile,cs.year,cs.month,cs.city,cs.province,cs.popularity,cs.domain,cs.price,omd.price_bn,cs.status,cs.source_type,cs.expired_at,cs.sold_time from car_source as cs ' \
    #             'left join open_model_detail as omd on cs.model_detail_slug = omd.detail_model_slug ' \
    #             ' where cs.global_sibling = 0 and cs.model_detail_slug is not null and cs.pub_time >= \''+pub_time+'\' '

    query_sql = 'select cs.pub_time,cs.id,cs.title,cs.mile,cs.year,cs.month,cs.province,cs.domain,cs.price from car_source as cs ' \
                ' where cs.global_sibling = 0  and domain in (\'guazi.com\',\'renrenche.com\',\'xin.com\') and cs.pub_time >= \''+pub_time+'\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def query_produce_open_model_detail():
    """
    查询款型库
    """
    query_sql = 'select id,price_bn,global_slug,year,volume,control,emission_standard,detail_model,detail_model_slug,status from open_model_detail where status = \'Y\' or status = \'A\' '

    engine = create_engine(gl.PRODUCE_PINGJIA_ENGINE, encoding=gl.ENCODING)
    return pd.read_sql_query(query_sql, engine)


def insert_or_update_base_standard_open_category(data):
    """
    查询款型库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    columns_update = [column_name + '=' + 'VALUES(' + column_name + ')' for column_name in list(data.columns)]
    columns_update = str(columns_update).replace('\'', '')
    columns_update = columns_update[1:len(columns_update) - 1]

    columns_name = str(list(data.columns)[1:])
    columns_name = columns_name[1:len(columns_name) - 1]
    columns_name = columns_name.replace('\'', '')
    with engine.begin() as con:
        for i in range(0, len(data)):
            if str(data.loc[i, 'id']) == 'nan':
                value = str([v if str(v) != 'nan' else 'null' for v in list(data.loc[i, :].values)][1:])
                value = value[1:len(value) - 1]
                value = re.sub(r'\'null\'', 'null', value)
                sql = 'INSERT INTO china_used_car_estimate.base_standard_open_category (' + columns_name + ') VALUES (' + value + ')'
            else:
                value = str(list(data.loc[i, :].values))
                value = value[1:len(value) - 1]
                value = re.sub(r' nan', ' null', value)
                sql = 'INSERT INTO china_used_car_estimate.base_standard_open_category VALUES (' + value + ') ON DUPLICATE KEY UPDATE ' + columns_update
            print(sql)
            con.execute(sql)
    con.close()


def insert_or_update_base_standard_open_model_detail(data):
    """
    查询款型库
    """
    engine = create_engine(gl.TEST_PINGJIA_ENGINE, encoding=gl.ENCODING)

    columns_update = [column_name + '=' + 'VALUES(' + column_name + ')' for column_name in list(data.columns)]
    columns_update = str(columns_update).replace('\'', '')
    columns_update = columns_update[1:len(columns_update) - 1]

    columns_name = str(list(data.columns)[1:])
    columns_name = columns_name[1:len(columns_name) - 1]
    columns_name = columns_name.replace('\'', '')
    with engine.begin() as con:
        for i in range(0, len(data)):
            if str(data.loc[i, 'id']) == 'nan':
                value = str([v if str(v) != 'nan' else 'null' for v in list(data.loc[i, :].values)][1:])
                value = value[1:len(value) - 1]
                value = re.sub(r'\'null\'', 'null', value)
                sql = 'INSERT INTO china_used_car_estimate.base_standard_open_model_detail (' + columns_name + ') VALUES (' + value + ')'
            else:
                value = str(list(data.loc[i, :].values))
                value = value[1:len(value) - 1]
                value = re.sub(r' nan', ' null', value)
                sql = 'INSERT INTO china_used_car_estimate.base_standard_open_model_detail VALUES (' + value + ') ON DUPLICATE KEY UPDATE ' + columns_update
            print(sql)
            con.execute(sql)
    con.close()
