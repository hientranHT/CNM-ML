import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None

rcParams['figure.figsize'] = 20, 10

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv("NSE-TATA.csv")
df.head()

df = df[["Close"]].copy()
df["target"] = df.Close.shift(-1)
df.dropna(inplace=True)


def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


train, test = train_test_split(df, 0.2)

X = train[:, :-1]
y = train[:, -1]

from xgboost import XGBRegressor

model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)

def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.fit(X, y)
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]

from sklearn.metrics import mean_squared_error
def validate(data, perc):
    predictions = []

    train, test = train_test_split(data, perc)

    history = [x for x in train]

    for i in range(len(test)):
        test_X, test_y = test[i, :-1], test[i, -1]

        pred = xgb_predict(history, test_X[0])
        predictions.append(pred)

        history.append(test[i])

    error = mean_squared_error(test[:, -1], predictions, squared=False)

    return error, test[:, -1], predictions

rmse, y, pred = validate(df, 0.2)



