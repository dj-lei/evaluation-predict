from evaluation.exception import *


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


class SqlOperateError(Exception):
    """
    操作数据库异常
    """
    def __init__(self, model_slug, message):
        self.error_type = gl.ERROR_SQL
        self.model_slug = model_slug
        self.message = message


