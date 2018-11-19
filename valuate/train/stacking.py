class Stacking(object):
    """
    组合模型训练
    """
    def __init__(self):
        self.x_all = []
        self.y_all = []

    def train_level2_model(self):
        """
        stack level2 模型训练
        """
        file_name = path + 'predict/model/model/xgb_stacking.model'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        d_train = xgb.DMatrix(self.x_all, label=self.y_all)
        model = xgb.train(als.xgb_level2_params, d_train, als.NUM_BOOST_ROUND)
        model.save_model(file_name)

    def execute(self):
        """
        执行多车型模型训练流程
        """
        try:
            # 训练估值预测模型
            data = pd.read_csv(path+'predict/model/data/train.csv')
            self.x_all = data.loc[:, gl.TRAIN_FEATURE]
            self.y_all = np.log(data[gl.TARGET_FEATURE])
            self.train_level2_model()
        except Exception:
            pass