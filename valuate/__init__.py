import os
import gc
import re
import ast
import json
import time
import math
import datetime
import traceback
import itertools
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")
# 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# 重定位根路径
from valuate.conf import global_settings as gl

path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

# from valuate.exception.api_error import SqlOperateError
# from valuate.exception.api_error import ApiParamsValueError
# from valuate.exception.api_error import ApiParamsTypeError
# from valuate.exception.api_error import ModelSlugPredictModelError
# from valuate.exception.api_error import ModelSlugPredictRecordError
# from valuate.exception.api_error import ModelSlugFeatureEngineeringError
# from valuate.exception.api_error import ModelSlugTrainError
# from valuate.exception.statistics_except import StatisticsExcept

from statistics import median
from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.optimize import leastsq

from valuate.db import db_operate
from valuate.db import process_tables

from valuate.process.process import Process


