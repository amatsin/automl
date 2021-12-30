import os
import shutil
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from wandb.xgboost import wandb_callback

import wandb
from dataloader import load_data
from imblearn.over_sampling import RandomOverSampler


def run():
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
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
    sampler = RandomOverSampler(random_state=RANDOM_STATE)

    oof_preds = np.zeros((len(train), 1))
    test_preds = np.zeros((len(test), 1))
    params = {
        "alpha": 0.6763306576868031,
        "colsample_bytree": 0.9781695650765355,
        "eta": 0.08258822790738427,
        "gamma": 0.5522919248987054,
        "max_depth": 3,
        "min_child_weight": 6,
        "subsample": 0.8225789960827787,
        "eval_metric": "auc",
    }
    ITERATIONS = 1000
    EARLY_STOP = 10
    config = dict(
        early_stop=EARLY_STOP,
        iterations=ITERATIONS,
        model="XGBoost",
        nfolds=NFOLDS,
    )
    wandb.init(
        project="baseline",
        entity="automldudes",
        config=config,
    )

    for fold_, (train_index, valid_index) in enumerate(folds.split(y, y)):
        print("Current Fold: {}".format(fold_))

        X_res, y_res = sampler.fit_resample(X[train_index, :], y[train_index])
        print(f"Training target statistics: {Counter(y_res)}")

        X_val, y_val = X[valid_index, :], y[valid_index]

        xg_train = xgb.DMatrix(X_res, y_res)
        xg_valid = xgb.DMatrix(X_val, y_val)

        clf = xgb.train(params, xg_train, ITERATIONS, evals=[(xg_train, "train"), (xg_valid, "eval")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False, callbacks=[wandb_callback()])

        val_pred = clf.predict(xgb.DMatrix(X[valid_index, :]))
        test_fold_pred = clf.predict(xgb.DMatrix(X_test))

        print("AUC = {}".format(metrics.roc_auc_score(y[valid_index], val_pred)))
        oof_preds[valid_index, :] = val_pred.reshape((-1, 1))
        test_preds += test_fold_pred.reshape((-1, 1))
    test_preds /= NFOLDS
    roc_score = metrics.roc_auc_score(y, oof_preds.ravel())
    print("Overall AUC = {}".format(roc_score))

    print("Saving submission file")
    sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
    sample.target = test_preds.astype(float)
    sample.ID_code = test_ids
    sample.to_csv('../model_predictions/submission_{}__{}.csv'.format(MODEL_NAME,
                                                                      str(roc_score)), index=False)

    print("Saving OOF predictions")
    oof_preds = pd.DataFrame(np.column_stack((train_ids,
                                              oof_preds.ravel())), columns=['ID_code', 'target'])
    oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(roc_score)), index=False)
    print("Saving code to reproduce")
    shutil.copyfile(os.path.basename(__file__),
                    '../model_source/{}__{}.py'.format(MODEL_NAME, str(roc_score)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    run()
