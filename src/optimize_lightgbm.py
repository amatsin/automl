import warnings

import numpy as np
from hyperopt import hp
from hyperopt import tpe, Trials
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    # Tuning guides:
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    # https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

    lgb_reg_params = {
        'n_estimators':  hp.choice('n_estimators', np.arange(100, 10000, 300)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'num_leaves': hp.choice('num_leaves', np.arange(20, 3000, 20)),
        'max_depth': hp.choice('max_depth', np.arange(3, 12, 1, dtype=int)),
        "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(100, 10000, 300)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)), # Why 1 to 8 while the suggested default is 0.01?
        'max_bin': hp.choice('max_bin', np.arange(200, 300, 5, dtype=int)),
        'lambda_l1': hp.choice('lambda_l1', np.arange(0, 100, 5)),
        'lambda_l2': hp.choice('lambda_l2', np.arange(0, 100, 5)),
        'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 15),
        'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.2, 0.95, 0.05)),
        'bagging_freq': hp.choice('bagging_freq', [1]),
        'feature_fraction': hp.choice('feature_fraction', np.arange(0.2, 0.95, 0.05))
    }

    lgb_fit_params = {} # seems that optimize.py is not using this dict for lgbm
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)

    return lgb_para


def optimize():
    lgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='lightgbm', space=lgb_para)
    lgb_opt, trials = obj.process(algo=tpe.suggest, max_evals=1000)
    print(lgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    optimize()
