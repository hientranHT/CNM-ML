import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None

rcParams['figure.figsize'] = 20, 10

scaler = MinMaxScaler(feature_range=(0, 1))

data = pd.read_csv("NSE-TATA.csv")

data["Date"] = pd.to_datetime(data.Date, format="%Y-%m-%d")
data.index = data['Date']

plt.figure(figsize=(16, 8))
plt.plot(data["Close"], label='Close Price history using XGBoost')
plt.title("Close Price history using XGBoost")


df = data.sort_index(ascending=True, axis=0)


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

model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
model.fit(X, y)



def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor()
    model.fit(X, y)
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]


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


n = int(len(df) * (1 - 0.2))
plt_valid_data = df[n:]
plt_valid_data['Predictions'] = pred
plt.plot(plt_valid_data[['Close', "Predictions"]])
train_data = df[:n]
plt.plot(train_data["Close"])


