from flask import Flask, render_template
from scripts.gen_algo import generator
from scripts.eq_odds import equalized_odds
from scripts.shap_api import drawBokehGraph, calibrated_equalized_odds_shap
import plotly
from bokeh.embed import components
from plotly.io import to_json
import plotly.graph_objs as go
import random

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

def genalgo_sc(dfs):
    fig = go.Figure()
    
    gens = [*range(0, 25, 1)]

    for i in gens:
        fig.add_trace(
            go.Scatter3d(
                x = dfs[i]['age'],
                y = dfs[i]['priors_count'],
                z = dfs[i]['juv_fel_count'],
                mode='markers'
            )
        )
    print("added traces")
    def create_button(gen):
        return dict(
            label = str(gen),
            method = 'update',
            args = [{'visible': dfs[gen].to_json,
                    'title': "gen " + str(gen)
            }]
        )

    fig.update_layout(updatemenus = [go.layout.Updatemenu(active=0, buttons=list(create_button(gen) for gen in gens))])
    print(type(fig))
    #fig_json = to_json(fig)
    return fig.show()

    #return json.dumps(genalgo, cls=plotly.utils.PlotlyJSONEncoder)
    

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
    ga_data = generator(random.randint(0,102))
    #print(type(ga_data[0]))
    scatter = ga_scatter(ga_data[0])
    #scatter = genalgo_sc(ga_data)
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

@app.route('/info/burden')
def burdeninfo():
    return render_template('burdeninfo.html')

@app.route('/shap-demo')
def shapdemo():
    dataset = 'data/even_better_shap_adult.csv'
    print('loaded data')
    ceq_g0_shap, ceq_g1_shap = calibrated_equalized_odds_shap(dataset, 'weighted', shap_enabled=True)
    ceq_g0_test, ceq_g1_test = calibrated_equalized_odds_shap(dataset, 'weighted', shap_enabled=False)
    print('did eq odds')
    shap_scatter = drawBokehGraph(dataset, ceq_g0_shap, ceq_g1_shap, ceq_g0_test, ceq_g1_test)
    script, div = components(shap_scatter)
    return render_template('shap-demo.html', shap_moh_live=shap_scatter, shap_div=div)

@app.route('/shapdemo')
def shapdem():
    return render_template('equalized.html')

@app.route('/burden')
def burdendemo():
    return render_template('burden.html')

if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=80)

