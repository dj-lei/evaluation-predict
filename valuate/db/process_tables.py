from valuate.db import *


def store_train_relative_data():
    """
    查询训练数据并存储在tmp中
    """
    try:
        car_source = db_operate.query_produce_car_source()
        car_source.to_csv(path+'../tmp/train/car_source.csv', index=False)
        del car_source
        gc.collect()

        # competitor_data = db_operate.query_produce_competitor_data()
        # competitor_data.to_csv(path+'../tmp/train/competitor_data.csv', index=False)
        # del competitor_data
        # gc.collect()

        # open_model_detail = db_operate.query_produce_open_model_detail()
        # open_model_detail = open_model_detail.rename(columns={'detail_model_slug': 'model_detail_slug'})
        # open_model_detail.to_csv(path+'../tmp/train/open_model_detail.csv', index=False)
        # del open_model_detail
        # gc.collect()
        #
        # open_category = db_operate.query_produce_open_category()
        # open_category.to_csv(path+'../tmp/train/open_category.csv', index=False)
        # del open_category
        # gc.collect()
        #
        # open_province_popularity = db_operate.query_produce_open_province_popularity()
        # open_province_popularity.to_csv(path+'../tmp/train/open_province_popularity.csv', index=False)
        # del open_province_popularity
        # gc.collect()
        #
        # open_city = db_operate.query_produce_open_city()
        # open_city.to_csv(path+'../tmp/train/open_city.csv', index=False)
        # del open_city
        # gc.collect()
        #
        # open_city = db_operate.query_produce_open_4s_price_clear()
        # open_city.to_csv(path+'../tmp/train/open_4s_price_clear.csv', index=False)
        # del open_city
        # gc.collect()
        print('下载训练相关表数据,已完成!')
    except Exception:
        print(traceback.format_exc())
        # raise SqlOperateError(gl.ERROR_SQL_DESC_QUERY, traceback.format_exc())


def store_models_divinable():
    """
    存储可预测车型款型到数据库
    """
    try:
        # 查询可预测款型
        details = db_operate.query_valuate_predict_data_divinable()
        model_detail_map = pd.read_csv(path + 'predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['model_detail_slug_id','model_slug','detail_model','model_detail_slug','price_bn','year','model_slug_id','brand_name','model_name','create_time']]
        model_detail_map = model_detail_map.loc[(model_detail_map['model_detail_slug_id'].isin(list(set(details.model_detail_slug_id.values)))), :]
        db_operate.insert_valuate_model_detail_map(model_detail_map)
    except Exception:
        raise SqlOperateError(gl.ERROR_SQL_DESC_INSERT, traceback.format_exc())


def query_model_source_data(start_time, model_slug):
    """
    查询指定车型原始数据
    """
    try:
        # car_source_history_tables = pd.read_csv(path+'../tmp/train/car_source_history_tables.csv')
        # tables = list(set(car_source_history_tables.table_name.values))

        reslut = pd.DataFrame()
        # 查询历史成交记录
        # for table in tables:
        #     data = db_operate.query_model_history_review_source_data(start_time, table, model_slug)
        #     reslut = reslut.append(data)
        # 查询生产成交记录
        data = db_operate.query_model_product_review_source_data(start_time, model_slug)
        reslut = reslut.append(data)
        # 查询生产最近三个月在售记录
        pub_time = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        time.sleep(0.2)
        data = db_operate.query_model_product_sale_source_data(pub_time, model_slug)
        reslut = reslut.append(data)
        os.makedirs(os.path.dirname(path + 'predict/model/train_source.csv'), exist_ok=True)
        reslut.to_csv(path + 'predict/model/train_source.csv', index=False)
    except Exception:
        raise SqlOperateError(model_slug, traceback.format_exc())


def store_relative_data_to_database():
    """
    存储训练数据到数据库
    """
    try:
        # 存储估值预测数据
        result = pd.read_csv(path + 'predict/model/data/result.csv')
        brand_slug_id = result.loc[0, 'brand_slug_id']
        db_operate.insert_valuate_predict_data(result, brand_slug_id)
    except Exception:
        raise SqlOperateError('test', traceback.format_exc())


def query_recently_review_car_source(start_time, end_time):
    """
    查询补充的训练数据
    """
    def cal_reg_year_month(df):
        """
        计算上牌年月
        """
        if len(str(df['reg_date'])) > 10:
            transaction_time = datetime.datetime.strptime(str(df['reg_date']), "%Y-%m-%d %H:%M:%S")
        else:
            transaction_time = datetime.datetime.strptime(str(df['reg_date']), "%Y-%m-%d")
        return pd.Series([transaction_time.year, transaction_time.month])

    try:
        add_data = db_operate.query_recently_review_car_source(start_time, end_time)
        add_c2b = db_operate.query_recently_review_deal_records(start_time, end_time)
        add_c2b = add_c2b.rename(columns={'source': 'domain', 'deal_date': 'sold_time'})
        add_c2b['expired_at'] = add_c2b['sold_time']
        add_c2b['url'] = 'N'
        add_c2b['dealer_id'] = 0
        add_c2b['source_type'] = 'sell_dealer'
        add_c2b['status'] = 'review'
        add_c2b[['year', 'month']] = add_c2b.apply(cal_reg_year_month, axis=1)
        add_c2b['mile'] = add_c2b['mile'] / 10000
        add_c2b['price'] = add_c2b['price'] / 10000
        add_c2b = add_c2b.drop(['reg_date'], axis=1)
        add_data = add_data.append(add_c2b)
        add_data.to_csv(path+'../tmp/train/add_train_source.csv', index=False)
    except Exception:
        raise SqlOperateError(gl.ERROR_SQL_DESC_QUERY, traceback.format_exc())


def update_artificial_relative_table():
    """
    更新人造数据相关表改P为Y
    """
    try:
        db_operate.update_valuate_difference_data_after_train()
        db_operate.update_add_data()
    except Exception:
        raise SqlOperateError(gl.ERROR_SQL_DESC_UPDATE, traceback.format_exc())


def store_operation_record_to_database(model_slug, operation_type):
    """
    存储操作记录
    """
    try:
        db_operate.insert_operation_record(model_slug, operation_type)

    except Exception:
        raise SqlOperateError(gl.ERROR_SQL_DESC_INSERT, traceback.format_exc())


def store_difference_data_to_database(data):
    """
    存储训练数据到数据库
    """
    try:
        cur_time = datetime.datetime.now()

        # 组合字段
        add_train_source = pd.read_csv(path + '../tmp/train/add_train_source.csv')
        add_train_source = add_train_source.loc[:, ['id', 'year', 'month']]
        data = data.merge(add_train_source, how='left', on=['id'])

        model_detail_map = pd.read_csv(path + 'predict/map/model_detail_map.csv')
        model_detail_map = model_detail_map.loc[:, ['brand_name', 'model_name', 'detail_model', 'model_detail_slug']]
        data = data.merge(model_detail_map)
        data['process_status'] = 'A'
        data['create_time'] = cur_time
        # 插入数据库
        data = data.rename(columns={'id': 'car_id'})
        data = data.loc[:, gl.VALUATE_DIFFERENCE_DATA_FEATURE]
        db_operate.insert_valuate_difference_data(data)

    except Exception:
        raise SqlOperateError(gl.ERROR_SQL_DESC_INSERT, traceback.format_exc())


def query_sell_and_sale(model_detail_slug):
    """
    查询指定车型原始数据
    """
    try:
        car_source_history_tables = pd.read_csv(path+'../tmp/train/car_source_history_tables.csv')
        tables = list(set(car_source_history_tables.table_name.values))
        start_time = str(datetime.datetime.now().year - 1) + '-01-01'

        model_detail_map = pd.read_csv(path + 'predict/map/model_detail_map.csv')
        model_slug = model_detail_map.loc[(model_detail_map['model_detail_slug'] == model_detail_slug), 'model_slug'].values[0]
        reslut = pd.DataFrame()
        # 查询历史成交记录
        for table in tables:
            data = db_operate.query_model_history_review_source_data(start_time, table, model_slug)
            reslut = reslut.append(data)
        # 查询生产成交记录
        data = db_operate.query_model_product_review_source_data(start_time, model_slug)
        reslut = reslut.append(data)
        # 查询生产最近三个月在售记录
        pub_time = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        time.sleep(0.2)
        data = db_operate.query_model_product_sale_source_data(pub_time, model_slug)
        reslut = reslut.append(data)
        reslut = reslut.sort_values(['model_detail_slug', 'city', 'expired_at'], ascending=[True, True, False])
        reslut.reset_index(inplace=True, drop=True)

        model_detail_map = model_detail_map.loc[:, ['brand_name', 'model_name', 'detail_model', 'model_detail_slug']]
        reslut = reslut.merge(model_detail_map, how='left', on=['model_detail_slug'])
        return reslut
    except Exception:
        raise SqlOperateError(model_detail_slug, traceback.format_exc())




