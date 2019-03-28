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

import warnings
warnings.filterwarnings("ignore")
# 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# 重定位根路径
from evaluation.conf import global_settings as gl

path = os.path.abspath(os.path.dirname(gl.__file__))
path = path.replace('conf', '')

from evaluation.exception.api_error import ApiParamsValueError
from evaluation.exception.api_error import ApiParamsTypeError
from evaluation.exception.api_error import SqlOperateError

from statistics import median
from sqlalchemy import create_engine
from scipy.optimize import leastsq

from evaluation.db import db_operate
from evaluation.db import process_tables

from evaluation.predict.predict_local import PredictLocal
from evaluation.process.process import Process


