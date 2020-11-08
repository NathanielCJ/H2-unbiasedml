from flask import Flask, render_template
from scripts.gen_algo import generator
from scripts.eq_odds import equalized_odds
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

app = Flask(__name__)

def ga_scatter(df):
    fig = go.Figure()

    sc_dat = [
        go.Scatter3d(
            x = df['age'],
            y = df['priors_count'],
            z = df['juv_fel_count'],
            mode='markers'
        )        
    ]

    scatterJSON = json.dumps(sc_dat, cls=plotly.utils.PlotlyJSONEncoder)

    return scatterJSON

def create_plot():


    N = 40
    t = np.arange(0, 10, 0.5)
    x = np.cos(t)
    y = np.sin(t)
    df = pd.DataFrame({'x': x, 'y': y, 'z': t}) # creating a sample dataframe


    data = [
        go.Scatter3d(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y'],
            z=df['z']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def shap_plot():
    return None

@app.route('/')
def index():
    #return "Hello, World!"
    bar = create_plot()
    return render_template('base.html')
    #return render_template('index.html', plot=bar)

@app.route('/gen-algo')
def genalgo():
    #gens = np.arange(0, 25)
    ga_data = generator(0)
    #print(type(ga_data[0]))
    scatter = ga_scatter(ga_data[0])
    #bar = create_plot()
    return render_template('gen-algo.html', plot=scatter)
    #return render_template('gen-algo.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/info/shap')
def shap():
    return render_template('shap.html')

@app.route('/info/counterfactuals')
def counterfactuals():
    return render_template('counterfactuals.html')

@app.route('/info/burdeninfo')
def burdeninfo():
    return render_template('burdeninfo.html')

@app.route('/shap-demo')
def shapdemo():

    plot = shap_plot()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

