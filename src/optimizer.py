from copy import copy
from datetime import datetime

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from hyperopt import fmin, STATUS_OK
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from time import time

from dataloader import load_data
from monitor import Monitor


def prepare_data():
    train = load_data(load_test=False, n_train_rows=100) # set n_train_rows=100 for fast test iterations
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
        self.filename = f'{fn_name}_at_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        self.monitor = Monitor(self.baseline_loss, self.filename)

    def process(self, algo, max_evals, existing_trials_file=None):
        ts = time()
        result = fmin(
            fn=self.crossvalidate,
            space=self.space,
            algo=algo,
            max_evals=max_evals,
            early_stop_fn=self.monitor.inspect_result,
            trials_save_file=existing_trials_file if existing_trials_file else f'{self.filename}.bin'
        )
        te = time()
        print('hyperopt took: %2.4f sec' % (te - ts))
        return result

    def crossvalidate(self, para):
        folds = StratifiedKFold(n_splits=self.NFOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        oof_preds = np.zeros((len(self.X), 1))

        sampler = RandomOverSampler(random_state=self.RANDOM_STATE)

        for fold_, (train_index, valid_index) in enumerate(folds.split(self.y, self.y)):
            trn_x, trn_y = sampler.fit_resample(self.X[train_index, :], self.y[train_index])
            val_x, val_y = self.X[valid_index, :], self.y[valid_index]

            val_pred = self.fn(para, trn_x, trn_y, val_x, val_y)

            print(f"AUC = {metrics.roc_auc_score(val_y, val_pred)}")
            oof_preds[valid_index, :] = val_pred.reshape((-1, 1))

        loss = para['loss_func'](self.y, oof_preds.ravel())
        return {'loss': loss, 'status': STATUS_OK}

    def xgboost(self, para, trn_x, trn_y, val_x, val_y):
        trn_data = xgb.DMatrix(trn_x, trn_y)
        val_data = xgb.DMatrix(val_x, val_y)
        clf = xgb.train(para['reg_params'],
                        trn_data,
                        evals=[(trn_data, "train"), (val_data, "eval")],
                        **para['fit_params'])
        return clf.predict(xgb.DMatrix(val_x))

    def lightgbm(self, para, trn_x, trn_y, val_x, val_y):
        trn_data = lgb.Dataset(trn_x, trn_y, silent=True, params={'verbose': -1})
        val_data = lgb.Dataset(val_x, val_y, silent=True, params={'verbose': -1})
        num_round = 1000000
        early_stopping_rounds = 3500
        clf = lgb.train(para['reg_params'], trn_data, num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=False,
                        early_stopping_rounds=early_stopping_rounds)
        return clf.predict(val_x, num_iteration=clf.best_iteration)

    def find_baseline_loss(self):
        print('Finding baseline loss...')
        this_para = copy(self.space)
        this_para['reg_params'] = dict()
        baseline_loss = self.crossvalidate(this_para)['loss']
        print(f'Baseline loss: {baseline_loss}')
        return baseline_loss
