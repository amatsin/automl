import lightgbm as lgb
import xgboost as xgb
#import catboost as ctb
from hyperopt import fmin, STATUS_OK, STATUS_FAIL

# Source: https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
from sklearn.model_selection import train_test_split

from dataloader import load_data


class HyperBoostOptimizer(object):

    def __init__(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def prepare_data(self):
        train, test = load_data()

        y = train.target.values
        train = train.drop(['ID_code', 'target'], axis=1)
        X = train.values.astype(float)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    #def ctb_reg(self, para):
    #    reg = ctb.CatBoostRegressor(**para['reg_params'])
    #    return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}