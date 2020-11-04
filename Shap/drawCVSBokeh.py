def drawCSVBokeh():
  import numpy as np
  import pandas as pd
  from bokeh.plotting import figure, output_file, show
  from bokeh.models import LinearInterpolator, Span
  from bokeh.io import output_notebook
  from collections import Counter
  shap_df = pd.read_csv("shap_df.csv")
  random_shap_df = pd.read_csv("random_shap_df.csv")
  size_mapper=LinearInterpolator(
        x=[shap_df.weights.min(), shap_df.weights.max()],
        y=[5,50]
    )

  random_size_mapper=LinearInterpolator(
      x=[random_shap_df.weights.min(), random_shap_df.weights.max()],
      y=[5,50]
  )
  TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
  p = figure(tools=TOOLS)

  p.scatter("shap", "pred", source=shap_df, fill_color = 'blue', fill_alpha = 0.6, size = {'field':'weights', 'transform': size_mapper}, legend_label='shap based')
  p.scatter("shap", "pred", source=random_shap_df, fill_color = 'red', fill_alpha = 0.6, size = {'field':'weights', 'transform': size_mapper}, legend_label='randomized')
  p.xaxis.axis_label = "Shap Scores"
  p.yaxis.axis_label = "Predictions"
  vline = Span(location=0, dimension='height', line_color='red', line_width=3, line_dash="dotted")
  hline = Span(location=0.44, dimension='width', line_color='red', line_width=3,line_dash="dotted")

  p.renderers.extend([vline, hline])
  p.legend.click_policy="hide"
  show(p)  # open a browser