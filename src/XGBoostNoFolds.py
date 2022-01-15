import os
import shutil
import warnings
from collections import Counter

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler

from dataloader import load_data, load_test_data


def run():
    RANDOM_STATE = 42
    script_name = os.path.basename(__file__).split('.')[0]
    MODEL_NAME = "{0}".format(script_name)
    print("Model: {}".format(MODEL_NAME))
    train = load_data()
    test = load_test_data()
    y = train.target.values
    train = train.drop(['ID_code', 'target'], axis=1)
    feature_list = train.columns
    test_ids = test.ID_code.values
    test = test[feature_list]
    X = train.values.astype(float)
    X_test = test.values.astype(float)
    sampler = RandomOverSampler(random_state=RANDOM_STATE)

    params = {
        "alpha": 0.832486070007167,
        "colsample_bytree": 0.8892966997331828,
        "eta": 0.050656726987950894,
        "gamma": 0.0619160464029609,
        "max_depth": 2,
        "min_child_weight": 72,
        "subsample": 0.8247362099753516,
        'eval_metric':      'auc',
        'objective':        'binary:logistic',
        'tree_method':      'gpu_hist'
    }

    xgb_fit_params = {
        'num_boost_round': 4000,
        'early_stopping_rounds': 50,
        'verbose_eval': 100
    }

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"Training target statistics: {Counter(y_res)}")

    xg_train = xgb.DMatrix(X_res, y_res)

    clf = xgb.train(params, xg_train, evals=[(xg_train, "train")], **xgb_fit_params)
    test_preds = clf.predict(xgb.DMatrix(X_test))
    print("Saving submission file")
    sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
    sample.target = test_preds.astype(float)
    sample.ID_code = test_ids
    sample.to_csv('../model_predictions/submission_{}.csv'.format(MODEL_NAME), index=False)

    print("Saving code to reproduce")
    shutil.copyfile(os.path.basename(__file__), '../model_source/{}.py'.format(MODEL_NAME))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    run()
