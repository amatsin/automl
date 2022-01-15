import pickle

import numpy as np
import pandas as pd
from autofeat import AutoFeatRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def load_test_data(scale=True, autofeat_transform=False):
    print("Loading test...")
    original_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    print("Original test shape: ", original_test.shape)

    # splits into two, row-wise
    test_synth_removed, test_synth = remove_synthetic(original_test)

    # additional columns only for some rows
    freq_enc_test_features = frequency_encoding(test_synth_removed)

    if autofeat_transform:
        with open('autofeat_regressor.pickle', mode='rb') as fp:
            autofeat_transformer: AutoFeatRegressor = pickle.load(fp)
        with_original_and_new_features = autofeat_transformer.transform(
            original_test.drop(['ID_code'], axis=1).values.astype(float))
        test = pd.concat([with_original_and_new_features, original_test.ID_code], axis=1)
    else:
        test = original_test

    # merges new columns to original (or to original plus autofeat)
    test = pd.concat([test, freq_enc_test_features], axis=1)
    # empty values will be 0
    test.fillna(0, inplace=True)

    if scale:
        train = load_data(scale=False, autofeat_transform=autofeat_transform)
        test = scale_test_data(train, test)

    return test


def scale_test_data(train, test):
    print('Scaling data...')
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns
    X_test = test.drop(['ID_code'], axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    test[X_columns] = X_test
    return test


def remove_synthetic(test):
    # Code taken from: https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
    print("Removing synthetic data from test")
    df_test = test.drop(['ID_code'], axis=1, errors='ignore')

    df_test = df_test.values

    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    print(f'real_samples_indexes {len(real_samples_indexes)}')
    print(f'synthetic_samples_indexes {len(synthetic_samples_indexes)}')

    return test.iloc[~test.index.isin(synthetic_samples_indexes)], test.iloc[test.index.isin(synthetic_samples_indexes)]


def load_data(scale=True, n_train_rows=None, autofeat_transform=False):
    print("Reading training data...")
    original_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')[:n_train_rows]
    print("Original train shape: ", original_train.shape)

    freq_enc_train_features = frequency_encoding(original_train)

    if autofeat_transform:
        train = pd.concat([add_autofeat_features(original_train), freq_enc_train_features], axis=1)
    else:
        train = pd.concat([original_train, freq_enc_train_features], axis=1)

    if scale:
        train = scale_data(train)
    print('target data type is', train['target'].dtype)
    print('shape returned by load_data function', train.shape)
    return train


def add_autofeat_features(data):
    """
    Inputs:
        - X: pandas dataframe or numpy array with original features (n_datapoints x n_features)
    Returns:
        - new_df: new pandas dataframe with all the original features (except categorical features transformed
                  into multiple 0/1 columns) and the most promising engineered features.
    """
    with open('autofeat_regressor.pickle', mode='rb') as fp:
        autofeat_transformer: AutoFeatRegressor = pickle.load(fp)
    with_original_and_new_features = autofeat_transformer.transform(
        data.drop(['ID_code', 'target'], axis=1).values.astype(float))
    result = pd.concat([with_original_and_new_features, data.ID_code, data.target], axis=1)
    print('shape returned by add_autofeat_features function', result.shape)
    return result


def scale_data(train):
    print('Scaling data...')
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    train[X_columns] = X_train

    return train


def frequency_encoding(train):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv
    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    train_df = pd.DataFrame()
    for col in idx:
        train_df[col + '_freq'] = train[col].map(train.groupby(col).size())
        train_df = train_df.copy()

    print('shape returned by frequency_encoding function', train_df.shape)
    return train_df
