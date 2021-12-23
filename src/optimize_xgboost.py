import warnings

import numpy as np
from hyperopt import hp
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    xgb_reg_params = {
        # 'eta':              hp.choice('eta', [0.01, 0.015, 0.025, 0.05, 0.1]),
        # 'gamma':            hp.choice('gamma', np.concatenate((np.arange(0.05, 0.11, 0.01), [0.3, 0.5, 0.7, 0.9, 1.0]))),
        # 'max_depth':        hp.choice('max_depth', [3, 5, 7, 9, 12, 15, 17, 25]),
        # 'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
        # 'subsample':        hp.choice('subsample', np.arange(0.6, 1.0, 0.1)),
        # 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.6, 1.0, 0.1)),
        # 'lambda':           hp.choice('alpha', [0.0, 0.1, 0.5, 1.0]),

        'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
        'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'n_estimators':     100,
    }
    xgb_fit_params = {
        'eval_metric': 'auc',
        'early_stopping_rounds': 3500,
        'verbose': False
    }
    xgb_para = dict()
    xgb_para['reg_params'] = xgb_reg_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)
    return xgb_para


def optimize():
    xgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='xgb_clf', space=xgb_para)
    xgb_opt = obj.process()
    print(xgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    optimize()
