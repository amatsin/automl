import pickle

import numpy as np
import pandas as pd
from autofeat import AutoFeatRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def load_data(scale=True, load_test=True, n_train_rows=None, remove_synth=True, autofeat_transform=False):
    print("Reading training data...")
    original_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')[:n_train_rows]
    print("Original train length: ", len(original_train))

    if load_test:
        return load_with_test(remove_synth, scale, original_train, autofeat_transform=autofeat_transform)

    freq_enc_train_features = frequency_encoding_train(original_train)

    if autofeat_transform:
        train = pd.concat([add_autofeat_features(original_train), freq_enc_train_features], axis=1)
    else:
        train = pd.concat([original_train(original_train), freq_enc_train_features], axis=1)

    if scale:
        train = scale_data_train(train)

    # TODO: we should not need to do this
    train['target'] = train['target'].astype(int)

    return train


def load_with_test(remove_synth, scale, original_train, autofeat_transform=False):
    print("Loading test...")
    original_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    if remove_synth:
        test_synth_removed, test_synth = remove_synthetic(original_test)
        freq_enc_train_features, freq_enc_test_features = frequency_encoding(original_train, test_synth_removed, test_synth)
    else:
        freq_enc_train_features, freq_enc_test_features = frequency_encoding(original_train, original_test)

    if autofeat_transform:
        train = pd.concat([add_autofeat_features(original_train), freq_enc_train_features], axis=1)
        test = pd.concat([add_autofeat_features(original_test), freq_enc_test_features], axis=1)
    else:
        train = pd.concat([original_train, freq_enc_train_features], axis=1)
        test = pd.concat([original_test, freq_enc_test_features], axis=1)

    print("Test length: ", len(test))
    if scale:
        train, test = scale_data(train, test)

    # TODO: we should not need to do this
    train['target'] = train['target'].astype(int)

    return train, test


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
    return pd.concat([with_original_and_new_features, data.ID_code, data.target], axis=1)


def scale_data(train, test):
    print('Scaling data...')
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns
    X_test = test.drop(['ID_code'], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    print(X_train.shape)
    print(X_test.shape)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train[X_columns] = X_train
    test[X_columns] = X_test

    return train, test


def scale_data_train(train):
    print('Scaling data...')
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    train[X_columns] = X_train

    return train


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


def frequency_encoding(train, test, synth_rows=[]):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv

    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    traintest = pd.concat([train, test])
    traintest = traintest.reset_index(drop=True)

    traintest_new = pd.DataFrame()
    for col in idx:
        traintest_new[col + '_freq'] = traintest[col].map(traintest.groupby(col).size())

    train_df = traintest_new[:len(train)]
    test_df = traintest_new[len(train):]
    test_df = test_df.drop(['target'], axis=1, errors='ignore')

    if len(synth_rows):
        test_df = pd.concat([synth_rows, test_df], axis=0, ignore_index=True)
        test_df = test_df.fillna(0)

    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df


def frequency_encoding_train(train):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv
    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    train_df = pd.DataFrame()
    for col in idx:
        train_df[col + '_freq'] = train[col].map(train.groupby(col).size())
        train_df = train_df.copy()

    print('Only frequency encoding features df shape:', train_df.shape)
    return train_df
