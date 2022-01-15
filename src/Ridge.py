import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from dataloader import load_data

NFOLDS = 5
RANDOM_STATE = 42

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

print("Model: {}".format(MODEL_NAME))

train = load_data()

y = train.target.values
train_ids = train.ID_code.values
train = train.drop(['ID_code', 'target'], axis=1)
feature_list = train.columns

X = train.values.astype(float)

clfs = []
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train), 1))

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_, :], y[trn_]
    val_x, val_y = X[val_, :], y[val_]

    clf = RidgeClassifier(random_state=RANDOM_STATE).fit(trn_x, trn_y)
    val_pred = clf.decision_function(val_x)

    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))

roc_score = metrics.roc_auc_score(y, oof_preds.ravel())
print("Overall AUC = {}".format(roc_score))

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_ids, oof_preds.ravel())), columns=['ID_code', 'target'])
Path('../kfolds').mkdir(exist_ok=True)
oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(roc_score)), index=False)

print("Saving code to reproduce")
shutil.copyfile(os.path.basename(__file__), '../model_source/{}__{}.py'.format(MODEL_NAME, str(roc_score)))