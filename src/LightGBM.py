import os
import shutil
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import wandb
from dataloader import load_data

warnings.filterwarnings("ignore")  # to avoid deprecation warnings


NFOLDS = 5
RANDOM_STATE = 42

script_name = os.path.basename(__file__).split('.')[0]

MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

print("Model: {}".format(MODEL_NAME))

train, test = load_data()

y = train.target.values
train_ids = train.ID_code.values
train = train.drop(['ID_code', 'target'], axis=1)
feature_list = train.columns

test_ids = test.ID_code.values
test = test[feature_list]

X = train.values.astype(float)
X_test = test.values.astype(float)

clfs = []
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))

param = {'metric': 'auc'}
num_round = 1000000
early_stopping_rounds = 3500

config = dict(
    early_stop=early_stopping_rounds,
    iterations=num_round,
    model="LightGBM",
    nfolds=NFOLDS,
)

wandb.init(
    project="baseline",
    entity="automldudes",
    config=config,
)

sampler = RandomOverSampler(random_state=RANDOM_STATE)

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))

    trn_x_up, trn_y_up = sampler.fit_resample(X[trn_, :], y[trn_])
    val_x, val_y = X[val_, :], y[val_]

    trn_data = lgb.Dataset(trn_x_up, trn_y_up)
    val_data = lgb.Dataset(val_x, val_y)

    clf = lgb.train(param, trn_data, num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds=early_stopping_rounds,
                    callbacks=[wandb.lightgbm.wandb_callback()])

    val_pred = clf.predict(val_x, num_iteration=clf.best_iteration)
    test_fold_pred = clf.predict(X_test, num_iteration=clf.best_iteration)

    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))
    test_preds += test_fold_pred.reshape((-1, 1))

test_preds /= NFOLDS

roc_score = metrics.roc_auc_score(y, oof_preds.ravel())
print("Overall AUC = {}".format(roc_score))

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_ids,
                                          oof_preds.ravel())), columns=['ID_code', 'target'])
oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(roc_score)), index=False)

print("Saving code to reproduce")
shutil.copyfile(os.path.basename(__file__),
                '../model_source/{}__{}.py'.format(MODEL_NAME, str(roc_score)))

print("Saving submission file")
sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
sample.target = test_preds.astype(float)
sample.ID_code = test_ids
sample.to_csv('../model_predictions/submission_{}__{}.csv'.format(MODEL_NAME,
                                                                  str(roc_score)), index=False)
