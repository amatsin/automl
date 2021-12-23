import warnings

from sklearn.metrics import roc_auc_score

from optimizer import HyperBoostOptimizer


def parameters():
    lgb_best_params = {
        'bagging_fraction': 0.13,
        'bagging_freq': 0,
        'feature_fraction': 0.9,
        'lambda_l1': 0.5,
        'lambda_l2': 0.9,
        'learning_rate': 0.01122217943044039,
        'max_bin': 3,
        'max_depth': 1,
        'min_child_weight': 2,
        'min_data_in_leaf': 17,
        'min_gain_to_split': 0.0017854325018302752,
        'n_estimators': 22,
        'num_leaves': 11
    }

    lgb_fit_params = {
        'eval_metric': 'auc',
        'early_stopping_rounds': 350,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_best_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func'] = lambda y, pred: 1 - roc_auc_score(y, pred)

    return lgb_para


def run_once():
    lgb_para = parameters()
    obj = HyperBoostOptimizer(fn_name='lgb_clf', space=lgb_para)
    lgb_opt = obj.process()
    print(lgb_opt)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    run_once()
