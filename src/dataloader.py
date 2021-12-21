import pandas as pd
from sklearn.preprocessing import StandardScaler

def balance_data(train):
    # TODO: is there better way to sample?
    train_true = train.loc[train['target'] == 1]
    print("Train target true length: ", len(train_true))
    train_false = train.loc[train['target'] != 1].sample(frac=1)[:len(train_true)]
    print("Train target false length: ", len(train_false))
    train = pd.concat([train_true, train_false], ignore_index=True).sample(frac=1)
    print("Train balanced length: ", len(train))
    return train

def scale_data(train, test):
    X_train = train.drop(['ID_code', 'target'], axis=1)
    X_columns = X_train.columns
    X_test = test.drop(['ID_code'], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train[X_columns] = X_train
    test[X_columns] = X_test

    return train, test

def remove_synthetic(test):
    #Code taken from: https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
    print("Removing syntetic data from test")
    df_test = test
    df_test.drop(['ID_code'], axis=1, inplace=True)
    df_test = df_test.values

    unique_samples = []
    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    print(len(real_samples_indexes))
    print(len(synthetic_samples_indexes))
    
    return test
    
    
def frequency_encoding(train, test):
    
    
    return train, test


def load_data(scale=True):
    print("Reading training data")
    train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
    print("Train length: ", len(train))
    train = balance_data(train)
    test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
    test = remove_synthetic(test)
    print("Test length: ", len(test))
    if(scale):
        train, test = scale_data(train, test)
    
    return train, test
