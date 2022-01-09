import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm

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
    df_test = test.drop(['ID_code'], axis=1)

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


def frequency_encoding(train, test, synth_rows = []):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv

    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    traintest = pd.concat([train, test])
    traintest = traintest.reset_index(drop=True)

    for col in idx:
        traintest[col + '_freq'] = traintest[col].map(traintest.groupby(col).size())

    train_df = traintest[:len(train)]
    test_df = traintest[len(train):]
    test_df = test_df.drop(['target'], axis=1)

    if len(synth_rows):
        test_df = pd.concat([synth_rows, test_df], axis=0, ignore_index=True)
        test_df = test_df.fillna(0)

    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df

def frequency_encoding_train(train):
    print('Frequency encoding...')
    # Code taken from : https://www.kaggle.com/ilu000/simplistic-magic-lgbmv
    idx = [c for c in train.columns if c not in ['ID_code', 'target']]
    train_df = train.copy()
    for col in idx:
        train_df[col + '_freq'] = train[col].map(train.groupby(col).size())
        train_df = train_df.copy()

    print('Train shape:', train_df.shape)
    return train_df


def load_data(scale=True, load_test=True, n_train_rows=None, remove_synth=True):
    print("Reading training data...")
    train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')[:n_train_rows]
    print("Train length: ", len(train))

    if load_test:
        print("Loading test...")
        test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
        
        if remove_synth:
            test_synth_removed, test_synth = remove_synthetic(test)
            train, test = frequency_encoding(train, test_synth_removed, test_synth)
        else:
            train, test = frequency_encoding(train, test)
        
        print("Test length: ", len(test))
        if scale:
            train, test = scale_data(train, test)
        return train, test

    train = frequency_encoding_train(train)
    if scale:
        train = scale_data_train(train)
    return train
