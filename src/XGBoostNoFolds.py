import os
import shutil
import warnings
from collections import Counter

import pandas as pd
import wandb
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from wandb.xgboost import wandb_callback

from dataloader import load_data


def run():
    RANDOM_STATE = 42
    script_name = os.path.basename(__file__).split('.')[0]
    MODEL_NAME = "{0}".format(script_name)
    print("Model: {}".format(MODEL_NAME))
    train, test = load_data(remove_synth=False, n_train_rows=100)
    y = train.target.values
    train = train.drop(['ID_code', 'target'], axis=1)
    feature_list = train.columns
    test_ids = test.ID_code.values
    test = test[feature_list]
    X = train.values.astype(float)
    X_test = test.values.astype(float)
    sampler = RandomOverSampler(random_state=RANDOM_STATE)

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
        nfolds=1,
    )
    wandb.init(
        project="baseline",
        entity="automldudes",
        config=config,
    )

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"Training target statistics: {Counter(y_res)}")

    xg_train = xgb.DMatrix(X_res, y_res)

    clf = xgb.train(params, xg_train, ITERATIONS, evals=[(xg_train, "train")],
                    early_stopping_rounds=EARLY_STOP, verbose_eval=False, callbacks=[wandb_callback()])
    test_preds = clf.predict(xgb.DMatrix(X_test))
    print("Saving submission file")
    sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
    sample.target = test_preds.astype(float)
    sample.ID_code = test_ids
    sample.to_csv('../model_predictions/submission_{}.csv'.format(MODEL_NAME), index=False)

    print("Saving code to reproduce")
    shutil.copyfile(os.path.basename(__file__),
                    '../model_source/{}.py'.format(MODEL_NAME))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to avoid deprecation warning every iteration
    run()
