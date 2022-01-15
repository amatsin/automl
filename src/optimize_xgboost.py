import argparse
import warnings

from hyperopt import hp
from hyperopt import tpe
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    # Tuning guides:
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

    xgb_reg_params = {
        'eta':              hp.uniform('eta', 0.05, 0.5),
        'gamma':            hp.uniform('gamma', 0.05, 1.0),
        'lambda':           hp.uniform('alpha', 0.0, 1.0),
        'max_depth':        hp.uniformint('max_depth', 2, 25),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 75),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'subsample':        hp.uniform('subsample', 0.6, 1),
        'eval_metric':      'auc',
        'objective':        'binary:logistic',
        'tree_method':      'hist',
    }

    xgb_fit_params = {
        'num_boost_round': 4000,
        'early_stopping_rounds': 50,
        'verbose_eval': 100
    }

    xgb_para = dict()
    xgb_para['reg_params'] = xgb_reg_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)
    return xgb_para


def optimize(args: argparse.Namespace):
    xgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='xgboost', space=xgb_para, autofeat_transform=args.autofeat)
    xgb_opt = obj.process(algo=tpe.suggest, max_evals=1000)
    print(xgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--autofeat", type=bool, default=False)
    args = parser.parse_args()
    optimize(args)