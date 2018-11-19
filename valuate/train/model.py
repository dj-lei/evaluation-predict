from valuate import *


def xgb_train(x_all, y_all):
    """
    xgb回归模型训练
    """
    file_name = path+'predict/model/model/xgb_statistical.model'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    d_train = xgb.DMatrix(x_all, label=y_all)
    model = xgb.train(als.xgb_level2_params, d_train, als.NUM_BOOST_ROUND)
    model.save_model(file_name)


def xgb_brand_train(x_all, y_all):
    """
    xgb回归模型训练
    """
    file_name = path+'predict/model/model/xgb_stacking.model'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    d_train = xgb.DMatrix(x_all, label=y_all)
    model = xgb.train(als.xgb_level2_params, d_train, als.NUM_BOOST_ROUND)
    model.save_model(file_name)


def xgb_predict(data):
    """
    加载xgb模型
    """
    model = xgb.Booster()
    model.load_model(path+'predict/model/model/xgb_statistical.model')
    return pd.Series(np.exp(model.predict(xgb.DMatrix(data)))).values


def xgb_brand_predict(data):
    """
    加载xgb模型
    """
    model = xgb.Booster()
    model.load_model(path+'predict/model/model/xgb_stacking.model')
    return pd.Series(np.exp(model.predict(xgb.DMatrix(data)))).values