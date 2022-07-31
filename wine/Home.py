# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 09:26:14 2022

@author: alejo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

st.set_page_config(page_title = "Home page")



def plot_ts(DF, tooltip = False, rangeSelector = False, width = 600, height = 400):
  DF_Q = DF.select_dtypes(include = ['int64', 'float64'])
  DF_T = DF.select_dtypes(include = ['datetime64[ns]'])
  
  res = alt.Chart(DF).transform_fold(
    DF_Q.columns.values, as_ = ['series', 'value']
  ).mark_line().encode(
    x = DF_T.columns.values[0] + ':T', y = 'value:Q', color = 'series:N'
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



@st.cache(allow_output_mutation=True)
def load_data():
    growmark_locs = pd.read_csv("")
    return growmark_locs




st.sidebar.image('C:/Users/alejo/projects/wine-streamlit-demo/img/cropped-hires_vineyard_nutrition_logo_color-270x270.png', use_column_width=True)


        


st.title('Welcome to the Yield/Quality app')

st.markdown(
            """
            This app is part of the work of the HiRes Vineyard nutrition project, 
            \n 
            Please navigate this app by choosing the tab of interest on the sidebar of the screen 
            \n
            John, we´re obviously miles away from the perfect text here hehehee. But I guess the idea is 3 parragraphs:
                \n 
                1- Cordial welcome to the app and say what institutions and projects are behind it 
                \n 
                2- Very brief explanation of what this does: bascially take data from Oregon and make it useful to winemakers and people interested in wine in the PNW. 
                \n 
                3- Say what the sidebar is and that one can access the other tabs by choosing the right buttton. This will be self-evident from the app I think, but it never hurts to make it explicit.
            """)
        