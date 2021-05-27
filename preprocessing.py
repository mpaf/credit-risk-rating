import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'UCI_Credit_Card.csv')
    
    try:
        os.makedirs('/opt/ml/processing/output/train')
        os.makedirs('/opt/ml/processing/output/validation')
        os.makedirs('/opt/ml/processing/output/test')
        os.makedirs('/opt/ml/processing/output/baseline')
    except:
        pass
    
    print('Reading input data from {}'.format(input_data_path))
    
    data = pd.read_csv(input_data_path)

    data = data.drop(['ID'], axis=1)
    data.loc[data['EDUCATION'] > 3 , 'EDUCATION'] = 4
    data.loc[data['EDUCATION'] == 0 , 'EDUCATION'] = 4
    
    data.loc[data['MARRIAGE'] == 0 , 'MARRIAGE'] = 3

    columns_to_scale = ['AGE', 'LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    min_max_scaler = preprocessing.MinMaxScaler()

    data[columns_to_scale] = min_max_scaler.fit_transform(data[columns_to_scale])
    
    columns_to_onehot_encode = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    model_data = pd.get_dummies(data, prefix=columns_to_onehot_encode, columns=columns_to_onehot_encode)
 
    model_data = model_data[ ['default.payment.next.month'] + [ col for col in model_data.columns if col != 'default.payment.next.month' ] ]
    
    # Rename target column for model monitor to parse accurately
    model_data = model_data.rename(columns={"default.payment.next.month": "default_payment_next_month"})

    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10% 
            
    # Write the training/validation/test data as a CSV file without headers, with the target as the first column and with no row numbers
    train_data.to_csv('/opt/ml/processing/output/train/train.csv', header=False, index=False)
    validation_data.to_csv('/opt/ml/processing/output//validation/validation.csv', header=False, index=False)
    test_data.to_csv('/opt/ml/processing/output/test/test.csv', header=False, index=False)
    model_data.to_csv('/opt/ml/processing/output/baseline/processed_with_headers.csv', header=True, index=False)