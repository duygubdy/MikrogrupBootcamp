from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
from model import AttritionPerformancePredictor

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    predictor = AttritionPerformancePredictor('logistic_model_attrition.pkl', 'logistic_model_performance.pkl',
                                              'label_encoders_attrition.pkl')
    data = request.json
    attrition_prediction, performance_prediction = predictor.predict(data)
    return jsonify({'attrition_prediction': attrition_prediction.tolist(), 'performance_prediction': performance_prediction.tolist()})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

static_folder = os.path.join(app.root_path, 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

def grab_col_names(dataframe, cat_th=18, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def create_histograms_and_violin_plots(data, num_columns):
    histograms = []
    violin_plots = []

    for col in num_columns:
        histogram = go.Histogram(
            x=data[col],
            name=col,
            marker= dict(color='#8480f0')
        )
        histograms.append(histogram)

        violin_plot = go.Violin(
            y=data[col],
            name=col,
            marker= dict(color='#8f007d'),
            box_visible = True
        )
        violin_plots.append(violin_plot)

    return histograms, violin_plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model')
def show_model():
    return render_template('model.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    if request.form["att"]=="Upload":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filepath=filepath))
    else:
        return redirect(url_for('analyze', filepath=os.path.join('data','data.csv')))

@app.route('/analyze/<filepath>')
def analyze(filepath):
    print(filepath)
    data = pd.read_csv(filepath)

    summary = data.describe().to_html()

    cat_columns, num_columns, cat_but_car = grab_col_names(data)

    #Box
    box_plots = []
    for col in num_columns:
        box_plot = go.Box(
            y=data[col],
            name=col,
            marker=dict(color='#8480f0')
        )
        box_plots.append(box_plot)

    plot_div_box_plot= go.Figure(data=box_plots).update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#151159'),
         yaxis=dict(
        showgrid=True,
        gridcolor='gray'
    )
    ).to_html(full_html=False)

    histograms, violin_plots = create_histograms_and_violin_plots(data,num_columns)

    plot_div_histograms =[]
    for col,histogram in zip(num_columns, histograms):
        fig = go.Figure(data=[histogram])
        fig.update_layout(
            title={'text': f"{col}, 'x'.0.5"},
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
         yaxis=dict(
        showgrid=True,
        gridcolor='gray'
    )
        )
        plot_div_histograms.append(fig.to_html(full_html = False))

    plot_div_violin_plots = []

    for col, violin_plot in zip(num_columns, violin_plots):
        fig= go.Figure(data=[violin_plot])
        fig.update_layout(
            title={'text': f"{col}, 'x'.0.5"},
            xaxis={'visible': False},
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
         yaxis=dict(
        showgrid=True,
        gridcolor='gray'
    )
        )
        plot_div_violin_plots.append(fig.to_html(full_html = False))

    cat_analysis = {col: data[col].value_counts() for col in cat_columns}

    #kategorik
    cat_plots = {}
    for col in cat_columns:
        if isinstance(cat_analysis[col].index, np.ndarray):
            labels = cat_analysis[col].index.tolist()
        else:
            labels = cat_analysis[col].index.to_list()

        if isinstance(cat_analysis[col].values, np.ndarray):
            values = cat_analysis[col].values.tolist()
        else:
            values = cat_analysis[col].values.to_list()

        trace = {
            'x' : labels,
            'y' : values,
            'type': 'bar',
            'name' : col,
            'marker' : {
            'color' : 'rgb(255, 102, 102)',
            }
        }
        data1 = [trace]

        yaxis = {
            'gridcolor': 'grey',
            'gridwidth': 1,
        }

        layout = {
            'title' : f'{col}',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'font': {'color': 'black'},
            'yaxis' : yaxis,
        }

        cat_plots[col] = {'data': data1, 'layout': layout}

    return render_template('analyze.html', filepath=filepath, tables=[summary], plot_div_box_plot=plot_div_box_plot,
                           plot_div_histograms=plot_div_histograms, plot_div_violin_plots=plot_div_violin_plots,
                           cat_plots=cat_plots, num_columns=num_columns, cat_columns=cat_columns,
                           cat_analysis=cat_analysis, data=data)

if __name__ == '__main__':
        app.run(debug=True);
