from evaluation import *


# 返回结构格式
result_map = pd.DataFrame(columns=['intent', 'intent_source', 'predict_price'])
result_map['intent'] = pd.Series(gl.INTENT_TYPE)
result_map['intent_source'] = pd.Series(gl.INTENT_TYPE_CAN)