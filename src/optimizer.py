from copy import copy

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from hyperopt import fmin, STATUS_OK, STATUS_FAIL, tpe, Trials
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

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
        self.default_hyperparams_loss = self.find_loss_with_default_hyperparams()

    def early_stop_fn(self, trials, *args):
        last_result = trials.results[-1]
        if last_result['status'] != 'ok':
            return True, args
        if last_result['loss'] <= self.default_hyperparams_loss:
            print(f'Reached better or equal result {last_result["loss"]} than baseline {self.default_hyperparams_loss} in {len(trials.results)} trials')
            return True, args
        return False, args

    def process(self):
        np.random.seed(self.RANDOM_STATE)
        try:
            result = fmin(fn=self.fn, space=self.space, algo=tpe.suggest, trials=Trials(), early_stop_fn=self.early_stop_fn)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result

    def lgb_clf(self, para):
        clf = lgb.LGBMClassifier(**para['reg_params'])
        return self.crossvalidate(clf, para)

    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['reg_params'])
        return self.crossvalidate(clf, para)

    def crossvalidate(self, clf, para):
        folds = StratifiedKFold(n_splits=self.NFOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        oof_preds = np.zeros((len(self.X), 1))

        sampler = RandomOverSampler(random_state=self.RANDOM_STATE)

        for fold_, (train_index, valid_index) in enumerate(folds.split(self.y, self.y)):
            trn_x, trn_y = sampler.fit_resample(self.X[train_index, :], self.y[train_index])
            val_x, val_y = self.X[valid_index, :], self.y[valid_index]

            clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], **para['fit_params'])
            val_pred = clf.predict_proba(val_x)[:, 1]

            oof_preds[valid_index, :] = val_pred.reshape((-1, 1))

        loss = para['loss_func'](self.y, oof_preds.ravel())
        return {'loss': loss, 'status': STATUS_OK}

    def find_loss_with_default_hyperparams(self):
        this_para = copy(self.space)
        this_para['reg_params'] = dict()
        baseline_loss = self.fn(this_para)['loss']
        print(f'Baseline loss with default hyperparameters: {baseline_loss}')
        return baseline_loss
