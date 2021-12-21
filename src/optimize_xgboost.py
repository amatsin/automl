import numpy as np
from hyperopt import hp
from hyperopt import tpe, Trials
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    xgb_reg_params = {
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
    xgb_para['loss_func' ] = lambda y, pred: 1-roc_auc_score(y, pred)
    return xgb_para

def optimize():
    xgb_para = parameters()
    obj = HyperBoostOptimizer()
    xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
    print(xgb_opt)


if __name__ == "__main__":
    optimize()