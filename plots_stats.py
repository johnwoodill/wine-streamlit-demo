# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:35:40 2022

@author: alejo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from vega_datasets import data

source = data.cars()

## function

def plot_ts(DF, tooltip = False, rangeSelector = False, width = 600, height = 400):
  DF_Q = DF.select_dtypes(include = ['int64', 'float64'])
  DF_T = DF.select_dtypes(include = ['datetime64[ns]'])
  
  res = alt.Chart(DF).transform_fold(
    DF_Q.columns.values, as_ = ['series', 'valor']
  ).mark_line().encode(
    x = DF_T.columns.values[0] + ':T', y = 'valor:Q', color = 'series:N'
  ).properties(width = width, height = height)
  
  selection = alt.selection_single(
    fields = DF_T.columns.values.tolist(), nearest = True, 
    on = 'mouseover', empty = 'none', clear = 'mouseout'
  )
  
  rule = alt.Chart(DF).mark_rule().encode(
    x = DF_T.columns.values[0] + ':T',
    opacity = alt.condition(selection, alt.value(0.3), alt.value(0))
  ).add_selection(selection)
  
  if tooltip == True:
    points = res.mark_point().transform_filter(selection)
    
    rule = alt.Chart(DF).mark_rule().encode(
      x = DF_T.columns.values[0] + ':T',
      opacity = alt.condition(selection, alt.value(0.3), alt.value(0)),
      tooltip = np.append(DF_T.columns.values, DF_Q.columns.values).tolist()
    ).add_selection(selection)
    
    if rangeSelector == False:
      return(res + points + rule)
  
  if rangeSelector == True:
    brush = alt.selection(type = 'interval', encodings = ['x'])
    res = res.encode(
      alt.X(DF_T.columns.values[0] + ':T', scale = alt.Scale(domain = brush))
    )
    
    points = res.mark_point().transform_filter(selection)
    
    rango = alt.Chart(DF).mark_area().encode(
      x = alt.X(DF_T.columns.values[0] + ':T', title = ''), 
      y = alt.Y(DF_Q.columns.values[0] + ':Q', title = '', axis = alt.Axis(labels = False))
    ).add_selection(brush).properties(width = width, height = (height * 0.1))
    
    res = alt.vconcat(alt.layer(res, points, rule, points), rango)
  
  return(res)



rain_temp = pd.read_csv("C:/Users/alejo/projects/wine-streamlit-demo/rain_temp.csv")


fechas = pd.DatetimeIndex(rain_temp.Date)
ts = pd.Series(rain_temp.rains.values, index = fechas)
ts

ts.plot()
plt.ylabel('Rainfall in mm')
plt.xlabel('Year')
plt.show()


ts2 = pd.Series(rain_temp.temps.values, index = fechas)
ts2

ts2.plot()
plt.ylabel('Temperature, celsius')
plt.xlabel('Year')
plt.show()


ts_df = ts2.to_frame(name = "valor").reset_index()


ts_plot = plot_ts(ts_df)
ts_plot


DF_Q = ts_df.select_dtypes(include = ['int64', 'float64'])
DF_T = ts_df.select_dtypes(include = ['datetime64[ns]'])

res = alt.Chart(ts_df).transform_fold(
    DF_Q.columns.values, as_ = ['series', 'valor']
  ).mark_line().encode(
    x = DF_T.columns.values[0] + ':T', y = 'valor:Q', color = 'series:N'
  ).properties(width = width, height = height)



df = pd.read_csv("C:/Users/alejo/projects/wine-streamlit-demo/patty_2021.csv")

df2 = df

df2.loc[len(df2.index)] = [ 93]