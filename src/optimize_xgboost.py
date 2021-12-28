import warnings

import numpy as np
from hyperopt import hp
from hyperopt import tpe, Trials
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    # Tuning guides:
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    xgb_reg_params = {
        'eta':              hp.choice('eta', np.arange(0.05, 0.5, 0.05)),
        'gamma':            hp.choice('gamma', np.concatenate((np.arange(0.05, 0.11, 0.01), [0.3, 0.5, 0.7, 0.9, 1.0]))),
        'lambda':           hp.choice('alpha', [0.0, 0.1, 0.5, 1.0]),
        'max_depth':        hp.choice('max_depth', [3, 5, 7, 9, 12, 15, 17, 25]),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 9, 2, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 1.0, 0.1)),
        'subsample':        hp.uniform('subsample', 0.6, 1),
        'eval_metric': 'auc'
    }
    xgb_fit_params = {
        'num_boost_round': 1000,
        'early_stopping_rounds': 10,
        'verbose_eval': False
    }
    xgb_para = dict()
    xgb_para['reg_params'] = xgb_reg_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)
    return xgb_para


def optimize():
    xgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='xgboost', space=xgb_para)
    xgb_opt, trials = obj.process(trials=Trials(), algo=tpe.suggest, max_evals=100)
    print(xgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    optimize()
