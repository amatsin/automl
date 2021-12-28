import warnings

from hyperopt import hp
from hyperopt import tpe, Trials
from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    # Tuning guides:
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    # https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

    lgb_reg_params = {
        'n_estimators':  hp.uniformint('n_estimators', 100, 10000),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'num_leaves': hp.uniformint('num_leaves', 20, 3000),
        'max_depth': hp.uniformint('max_depth', 3, 12),
        "min_data_in_leaf": hp.uniformint('min_data_in_leaf', 100, 10000),
        'min_child_weight': hp.uniform('min_child_weight', 1, 8), # TODO: Default is actually 0.01 in https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html?highlight=LGBMClassifier
        'max_bin': hp.uniformint('max_bin', 200, 300),
        'lambda_l1': hp.uniform('lambda_l1', 0, 100),
        'lambda_l2': hp.uniform('lambda_l2', 0, 100),
        'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 15),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 0.95),
        'bagging_freq': hp.choice('bagging_freq', [1]),
        'feature_fraction': hp.uniform('feature_fraction', 0.2, 0.95)
    }

    lgb_fit_params = {
        'eval_metric': 'auc',
        'early_stopping_rounds': 350,
        'verbose': False,
        'verbose_eval': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)

    return lgb_para


def optimize():
    lgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='crossvalidate_lightgbm', space=lgb_para)
    lgb_opt, trials = obj.process(trials=Trials(), algo=tpe.suggest)
    print(lgb_opt)
    return trials


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    optimize()
