## Create environment
```bash
conda create -n santander -c conda-forge scikit-learn pandas six python=3.8
conda activate santander
```

## Download dataset

```bash
cd input
kaggle competitions download -c santander-customer-transaction-prediction
unzip santander-customer-transaction-prediction.zip -d santander-customer-transaction-prediction
rm santander-customer-transaction-prediction.zip
```

## Running model
```bash
cd src
python LinearRegression.py
```

## Submit model to Kaggle
```bash
kaggle competitions submit -c santander-customer-transaction-prediction -f submission_LogisticRegression__folds5__0.611460871943926.csv -m "LogisticRegression"
```