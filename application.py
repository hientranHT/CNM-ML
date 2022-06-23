from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import MetaTrader5 as mt5
from xgboost import XGBRegressor

from mt5_funcs import get_symbol_names, TIMEFRAMES, TIMEFRAME_DICT
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

symbol_dropdown = html.Div([
    html.P('Symbol:'),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in get_symbol_names()],
        value='EURUSD'
    )
])

timeframe_dropdown = html.Div([
    html.P('Timeframe:'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[{'label': timeframe, 'value': timeframe} for timeframe in TIMEFRAMES],
        value='D1'
    )
])

num_bars_input = html.Div([
    html.P('Number of Candles'),
    dbc.Input(id='num-bar-input', type='number', value='20')
])

model_predict = html.Div([
    html.P('Model predict price'),
    dcc.Dropdown(
        id='model-predict',
        options=['LSTM', 'RNN', 'XGB'],
        value='LSTM'
    )
])

type_predict = html.Div([
    html.P('Model predict price'),
    dcc.Dropdown(
        id='type-predict',
        options=['Closing', 'Price of Change'],
        value='Closing'
    )
])

# creates the layout of the App
app.layout = html.Div([
    html.H1('Web App Predict Stock Price And Real Time Charts'),

    dbc.Row([
        dbc.Col(symbol_dropdown),
        dbc.Col(timeframe_dropdown),
        dbc.Col(num_bars_input),
        dbc.Col(model_predict),
        dbc.Col(type_predict)
    ]),

    html.Hr(),

    dcc.Interval(id='update', interval=2000),

    html.Div(id='page-content')

], style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '20px'})


def train_test_split(data, perc):
    data = data.values
    n = int(len(data) * (1 - perc))
    return data[:n], data[n:]


def xgb_predict(train, val, model):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]

    model = model
    model = XGBRegressor()
    model.fit(X, y)
    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]


def validate(data, model):
    predictions = []
    train, test = train_test_split(data, 0)
    temp = train
    train, test = train_test_split(data, 1)
    history = [x for x in temp]
    for i in range(len(test)):
        test_X, test_y = test[i, :-1], test[i, -1]

        pred = xgb_predict(history, test_X[0], model)
        predictions.append(pred)

        history.append(test[i])

    return predictions


def predict(df, name_model_predict, name_type_predict):
    if name_type_predict == 'Price of Change':
        if (name_model_predict == 'LSTM'):
            # and name_type_predict == 'Closing'
            model = load_model("saved_model_POC.h5")

        if (name_model_predict == 'RNN'):
            model = load_model("saved_rnn_model_POC.h5")

        if (name_model_predict == 'XGB'):
            model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
            model.load_model("saved_XGB_model_POC.json")

        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['time', 'poc'])

        for i in range(0, len(data)):
            new_data["time"][i] = data['time'][i]
            new_data["poc"][i] = data["poc"][i]

        new_data.index = new_data.time
        new_data.drop("time", axis=1, inplace=True)
        new_data["poc"][0] = float("NaN")
        df.dropna(inplace=True)
        dataset = new_data.values

        if name_model_predict == 'XGB':
            data_xgb = df[["poc"]].copy()
            data_xgb["target"] = data_xgb.poc.shift(-1)
            data_xgb["target"][len(data_xgb) - 1] = data_xgb["poc"][0]
            pred = validate(data_xgb, model)

        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            inputs = new_data.values
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_test = np.array(inputs)

            pred = model.predict(X_test)
            pred = scaler.inverse_transform(pred)

        if (name_model_predict == 'RNN'):
            pred = pred * -1
        if (name_model_predict == 'RNN' or name_model_predict == 'LSTM'):
            avg = 0
            for i in range(1, len(pred)):
                avg += pred[i][0]
            avg = avg / (len(pred) - 1)
            for i in range(1, len(pred)):
                pred[i][0] = pred[i][0] - avg

        valid = new_data
        valid['Predictions'] = pred

        return valid, pred[len(pred) - 1]

    if (name_model_predict == 'LSTM'):
        # and name_type_predict == 'Closing'
        model = load_model("saved_model.h5")

    if (name_model_predict == 'RNN'):
        model = load_model("saved_rnn_model.h5")

    if (name_model_predict == 'XGB'):
        model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        model.load_model("saved_XGB_model.json")

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['time', 'close'])

    for i in range(0, len(data)):
        new_data["time"][i] = data['time'][i]
        new_data["close"][i] = data["close"][i]

    new_data.index = new_data.time
    new_data.drop("time", axis=1, inplace=True)
    dataset = new_data.values

    if name_model_predict == 'XGB':
        data_xgb = df[["close"]].copy()
        data_xgb["target"] = data_xgb.close.shift(-1)
        data_xgb["target"][len(data_xgb) - 1] = data_xgb["close"][0]
        pred = validate(data_xgb, model)

    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        inputs = new_data.values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = np.array(inputs)

        pred = model.predict(X_test)
        pred = scaler.inverse_transform(pred)

    valid = new_data
    valid['Predictions'] = pred

    return valid, pred[len(pred) - 1]


@app.callback(
    Output('page-content', 'children'),
    Input('update', 'n_intervals'),
    State('symbol-dropdown', 'value'), State('timeframe-dropdown', 'value'), State('num-bar-input', 'value'),
    State('model-predict', 'value'),
    State('type-predict', 'value')
)
def update_ohlc_chart(interval, symbol, timeframe, num_bars, model_predict, type_predict):
    timeframe_str = timeframe
    timeframe = TIMEFRAME_DICT[timeframe]
    num_bars = int(num_bars)
    name_model_predict = model_predict
    name_type_predict = type_predict

    print(symbol, timeframe, num_bars, name_model_predict, name_type_predict)

    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    temp = []
    temp.append(0)
    for i in range(1, len(df)):
        temp.append(((df["close"][i] / df["close"][i - 1]) - 1) * 100)

    df = df.assign(poc=temp)
    fig = go.Figure(data=go.Candlestick(x=df['time'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close']))

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(yaxis={'side': 'right'})
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    valid, price_predicted_next_timeframe = predict(df, name_model_predict, name_type_predict);

    return [
        html.H2(id='chart-details', children=f'{symbol} - {timeframe_str}'),
        dcc.Textarea(
            id='textarea-example',
            value='Predict price next timeframe: ' + str(price_predicted_next_timeframe),
            style={'width': '20%', 'height': 80, 'textAlign': 'center'},
        ),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        dcc.Graph(
            id="Predicted Data",
            figure={
                "data": [
                    go.Scatter(
                        x=valid.index,
                        y=valid["Predictions"],
                        mode='markers'
                    )

                ],
                "layout": go.Layout(
                    title='Predict price use scatter plot by ' + name_model_predict + ' model and ' + name_type_predict,
                    xaxis={'title': 'Date'},
                    yaxis={'title': name_type_predict + ' Rate'}
                )
            }
        )
    ]


if __name__ == '__main__':
    # starts the server
    app.run_server()
