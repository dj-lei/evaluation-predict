from valuate.exception import *


class ApiParamsValueError(Exception):
    """
    Api参数值异常
    """
    def __init__(self, name, value, message):
        self.error_type = gl.ERROR_PARAMS
        self.name = name
        self.value = value
        self.message = message


class ApiParamsTypeError(Exception):
    """
    Api参数类型异常
    """
    def __init__(self, name, value, message):
        self.error_type = gl.ERROR_PARAMS
        self.name = name
        self.value = value
        self.message = message


class ModelSlugFeatureEngineeringError(Exception):
    """
    车型特征工程异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_FE
        self.model_slug = model_slug
        self.message = message


class ModelSlugTrainError(Exception):
    """
    车型训练异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_TRAIN
        self.model_slug = model_slug
        self.message = message


class ModelSlugPredictModelError(Exception):
    """
    车型预测异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_PREDICT_MODEL
        self.model_slug = model_slug
        self.message = message


class ModelSlugPredictRecordError(Exception):
    """
    车型预测异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_PREDICT_RECORD
        self.model_slug = model_slug
        self.message = message


class SqlOperateError(Exception):
    """
    操作数据库异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_SQL
        self.model_slug = model_slug
        self.message = message


class ManualError(Exception):
    """
    操作数据库异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_MANUAL
        self.model_slug = model_slug
        self.message = message

