import os
import json
import datetime

import flask
from werkzeug.utils import secure_filename
from dateutil.relativedelta import relativedelta


from utils import read_format_data, static_inference, train, train_test_split, get_pdq

app = flask.Flask(__name__)


@app.route("/")  # methods=["POST", "GET"])
def home_page():
    return flask.render_template('index.html')


@app.route("/predict", methods=["POST"])
def train_predict():
    output_data = {}

    print(flask.request.form)
    print(flask.request.files)

    form_data = flask.request.form.to_dict(flat=True)

    # Save file
    input_f = flask.request.files['input_file']
    filename = secure_filename(os.path.basename(input_f.filename))
    input_f.save(filename)

    # Read File to DF
    df = read_format_data(filename)

    # Train Test Splits
    train_df, test_df = train_test_split(df)

    # Get p,d,q values from auto arima
    p, d, q = get_pdq(train_df)

    # Train Model on training data
    mtrain_fit = train(train_df, p, d, q)

    # Store training and eval stats
    test_data_predictions = static_inference(
        mtrain_fit, n_months=test_df.shape[0])

    output_data['train'] = {
        'train_months': train_df.index.strftime('%b-%y').values.tolist(),
        'train_actual_values': train_df['Price'].values.tolist()
    }

    output_data['test'] = {
        'test_months': test_df.index.strftime('%b-%y').values.tolist(),
        'test_actual_values': test_df['Price'].values.tolist(),
        'test_prediction_values': test_data_predictions.tolist()
    }

    output_data['input_data'] = {
        'months': df.index.strftime('%b-%y').values.tolist(),
        'price_values': df['Price'].values.tolist()
    }

    num_forecastMonths = int(form_data['forecastMonth'])

    if form_data['forecast_method'] == "static":
        # Get p,d,q values from auto arima for entire df
        p, d, q = get_pdq(df)

        # Train Model on entire data
        mtrain_fit = train(df, p, d, q)

        # Fetch Future Predictions
        future_forecasts = static_inference(
            mtrain_fit, n_months=num_forecastMonths)

        future_months = []
        prev_month = df.iloc[-1].name
        for i in range(num_forecastMonths):
            prev_month = prev_month + relativedelta(months=1)
            future_months.append(datetime.datetime.strftime(
                prev_month, format='%b-%y'))

        output_data['predictions'] = {
            'pred_months': future_months,
            'pred_values': future_forecasts.tolist()
        }
    elif form_data['forecast_method'] == "rolling":

        pred_values = []
        future_months = []
        prev_month = df.iloc[-1].name

        print(num_forecastMonths)
        for i in range(num_forecastMonths):
            # Get p,d,q values from auto arima for entire df
            p, d, q = get_pdq(df)

            # Train Model on entire data
            mtrain_fit = train(df, p, d, q)

            # Fetch Future Predictions
            future_forecasts = static_inference(mtrain_fit, n_months=1)

            prev_month = prev_month + relativedelta(months=1)

            future_months.append(datetime.datetime.strftime(
                prev_month, format='%b-%y'))

            df.loc[prev_month] = future_forecasts[0]

            pred_values.append(future_forecasts[0])

            print("DF Shape: ", df.shape)

        output_data['predictions'] = {
            'pred_months': future_months,
            'pred_values': pred_values
        }

    print(output_data)

    os.system('rm ' + filename)

    return flask.Response(json.dumps(output_data), mimetype='application/json')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
