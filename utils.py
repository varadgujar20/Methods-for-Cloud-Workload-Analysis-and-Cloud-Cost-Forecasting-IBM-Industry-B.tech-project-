import re
import datetime
import calendar
from math import sqrt

import pandas as pd
import numpy as np
from pmdarima import auto_arima

from statsmodels.tsa.arima_model import ARIMA

def parse_date_string(x):
    dt_obj = datetime.datetime.strptime(x, "%b-%y")
    day=calendar.monthrange(dt_obj.year, dt_obj.month)[1]
    last_date = '{}'.format(day)+datetime.datetime.strftime(dt_obj, "-%m-%Y")
    return datetime.datetime.strptime(last_date, "%d-%m-%Y")

def read_format_data(filename):
    df = pd.read_csv(filename)

    #Change Months column from string to datetime format
    df['Months'] = df['Months'].apply(lambda x: parse_date_string(x))

    df.set_index('Months', inplace=True)

    #Change the Price Column to integers
    df['Price']=df['Price'].apply(lambda x:float(re.match(r'[0-9]*', x).group()))

    return df

def train_test_split(df, train_split_percentage = 0.85):
    n_obs = df.shape[0]

    train_data = df.iloc[:round(n_obs * train_split_percentage)]
    test_data = df.iloc[round(n_obs * train_split_percentage):]

    return train_data, test_data

def get_pdq(df):
    res = auto_arima(df['Price'], seasonal=False, trace=False)

    return res.to_dict()['order']

def train(df, p, d, q):
    model = ARIMA(df['Price'], order=(p,d,q))

    model_fit = model.fit()

    return model_fit

def static_inference(model_fit, n_months):
    # Forecast
    fc, se, conf = model_fit.forecast(steps=n_months)

    return fc

