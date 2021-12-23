from copy import copy

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from hyperopt import fmin, STATUS_OK, STATUS_FAIL
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from dataloader import load_data


def prepare_data():
    train, test = load_data()
    y = train.target.values
    train = train.drop(['ID_code', 'target'], axis=1)
    X = train.values.astype(float)
    return X, y


class HyperBoostOptimizer(object):
    NFOLDS = 5
    RANDOM_STATE = 42

    def __init__(self, fn_name, space):
        self.fn = getattr(self, fn_name)
        self.space = space
        self.X, self.y = prepare_data()
        self.baseline_loss = self.find_baseline_loss()

    def early_stop_fn(self, trials, *args):
        last_result = trials.results[-1]
        if last_result['status'] != 'ok':
            return True, args
        if last_result['loss'] < self.baseline_loss:
            print(f'Reached better result {last_result["loss"]} than baseline {self.baseline_loss} in {len(trials.results)} trials')
            return True, args
        return False, args

    def process(self, trials, algo):
        try:
            result = fmin(fn=self.fn, space=self.space, algo=algo, trials=trials, early_stop_fn=self.early_stop_fn)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result

    # todo: remove duplicated crossvalidation code
    def crossvalidate_xgboost(self, para):
        folds = StratifiedKFold(n_splits=self.NFOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        oof_preds = np.zeros((len(self.X), 1))

        sampler = RandomOverSampler(random_state=self.RANDOM_STATE)

        for fold_, (train_index, valid_index) in enumerate(folds.split(self.y, self.y)):
            trn_x, trn_y = sampler.fit_resample(self.X[train_index, :], self.y[train_index])
            val_x, val_y = self.X[valid_index, :], self.y[valid_index]

            xg_train = xgb.DMatrix(trn_x, trn_y)
            xg_valid = xgb.DMatrix(val_x, val_y)

            clf = xgb.train(para['reg_params'],
                            xg_train,
                            evals=[(xg_train, "train"), (xg_valid, "eval")],
                            **para['fit_params'])

            val_pred = clf.predict(xgb.DMatrix(val_x))

            print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
            oof_preds[valid_index, :] = val_pred.reshape((-1, 1))

        loss = para['loss_func'](self.y, oof_preds.ravel())
        return {'loss': loss, 'status': STATUS_OK}

    def crossvalidate_lighgbm(self, para):
        folds = StratifiedKFold(n_splits=self.NFOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        oof_preds = np.zeros((len(self.X), 1))

        sampler = RandomOverSampler(random_state=self.RANDOM_STATE)

        for fold_, (train_index, valid_index) in enumerate(folds.split(self.y, self.y)):
            trn_x, trn_y = sampler.fit_resample(self.X[train_index, :], self.y[train_index])
            val_x, val_y = self.X[valid_index, :], self.y[valid_index]

            trn_data = lgb.Dataset(trn_x, trn_y, silent=True, params={'verbose': -1})
            val_data = lgb.Dataset(val_x, val_y, silent=True, params={'verbose': -1})

            num_round = 1000000
            early_stopping_rounds = 3500

            clf = lgb.train(para['reg_params'], trn_data, num_round,
                            valid_sets=[trn_data, val_data],
                            verbose_eval=-1,
                            early_stopping_rounds=early_stopping_rounds)

            val_pred = clf.predict(val_x, num_iteration=clf.best_iteration)

            print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
            oof_preds[valid_index, :] = val_pred.reshape((-1, 1))

        loss = para['loss_func'](self.y, oof_preds.ravel())
        return {'loss': loss, 'status': STATUS_OK}

    def find_baseline_loss(self):
        this_para = copy(self.space)
        this_para['reg_params'] = dict()
        baseline_loss = self.fn(this_para)['loss']
        print(f'Baseline loss: {baseline_loss}')
        return baseline_loss
