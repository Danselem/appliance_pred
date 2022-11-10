#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score

import bentoml


## loading data
df = pd.read_csv("KAG_energydata_complete.csv")

df.columns = df.columns.str.lower()


lr = 0.1
depth = 10
n_splits = 5

output_file = f'model_lr={lr}.bin'



df['date'] = pd.to_datetime(df['date'])

def extract_dates(data=None, date_cols=None, subset=None, drop=True):
    df = data
    for date_col in date_cols:
        #Convert date feature to Pandas DateTime
        df[date_col ]= pd.to_datetime(df[date_col])

        #specify columns to return
        dict_dates = {  "dow":  df[date_col].dt.weekday,
                        "dom": df[date_col].dt.day,
                        "doy":   df[date_col].dt.dayofyear,
                        "hr": df[date_col].dt.hour,
                        "min":   df[date_col].dt.minute,
                        "is_wkd":  df[date_col].apply(lambda x : 1 if x  in [5,6] else 0 ),
                        "wkoyr": df[date_col].dt.isocalendar().week,
                        "mth": df[date_col].dt.month,
                        "qtr":  df[date_col].dt.quarter,
                        "yr": df[date_col].dt.year
                    } 

        if subset is None:
            #return all features
            subset = ['dow', 'dom', 'doy', 'hr', 'min', 'is_wkd', 'wkoyr', 'mth', 'qtr', 'yr']
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
        else:
            #Return only sepcified date features
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
                
    #Drops original time columns from the dataset
    if drop:
        df.drop(date_cols, axis=1, inplace=True)

    return df

df = extract_dates(data = df, date_cols = ['date'])


temp_params = ["t1", "t2", "t3", "t4", "t5", "t7", "t8",]

hum_params = ["rh_1", "rh_2", "rh_3", "rh_4", "rh_5", "rh_6", "rh_7", "rh_8", "rh_9"]

weather_params = ["t_out", "tdewpoint", "rh_out", "press_mm_hg", "windspeed", "visibility"]


date_params = ['date_dow', 'date_dom', 'date_doy', 'date_hr', 'date_min', 'date_wkoyr', 'date_mth', 'date_qtr',]

target = ["appliances"]


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# training 

def train(df_train, y_train, lr=0.1, depth=10):
    dicts = df_train[temp_params + hum_params + weather_params + date_params].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = GradientBoostingRegressor(learning_rate=lr, max_depth=depth, random_state=5)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[temp_params + hum_params + weather_params + date_params].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

# validation

print(f'doing validation with lr={lr}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=5)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log(df_train[target].values).ravel()
    y_val = np.log(df_val[target].values).ravel()

    dv, model = train(df_train, y_train, lr=lr, depth=depth)
    y_pred = predict(df_val, dv, model)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    scores.append(rmse)

    r2s = r2_score(y_val, y_pred)
    scores.append(r2s)

    print(f'rmse on fold {fold} is {rmse}')
    fold = fold + 1


print('validation results:')
print('depth=%s %.3f +- %.3f, %.3f +- %.3f' % (depth, np.mean(scores[0]), np.std(scores[0]), np.mean(scores[1]), np.std(scores[1])))


# training the final model

print('training the final model')

dv, model = train(df_full_train, np.log(df_full_train[target].values).ravel(), depth=depth, lr=lr)
y_pred = predict(df_test, dv, model)

y_test = np.log(df_test[target].values).ravel()

rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'rmse={rmse}')


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

# bentoml.sklearn.save_model(
#     'energy_predictor',
#     model,
#     custom_objects={
#         'dictVectorizer': dv
#     }, signatures = {
#     "predict": {
#         "batchable": True,
#         "batch_dim": 0,
#     }
#     })