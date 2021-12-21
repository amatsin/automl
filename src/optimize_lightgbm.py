import warnings

import numpy as np
from hyperopt import hp
from hyperopt import tpe, Trials
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    lgb_reg_params = {
        'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample': hp.uniform('subsample', 0.8, 1),
        'n_estimators': 100,
    }
    lgb_fit_params = {
        'eval_metric': 'auc',
        'early_stopping_rounds': 350,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)

    return lgb_para

def optimize():
    lgb_para = parameters()
    obj = HyperBoostOptimizer()
    lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
    print(lgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    optimize()