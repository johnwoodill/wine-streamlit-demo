# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:44:58 2022

@author: alejo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

df = pd.read_csv("./patty_reg.csv")

df2 = pd.read_csv("./patty_reg.csv")

df2.loc[len(df2.index)] = [999,0,df['LeafN'].mean() ,df['LeafP'].mean() ,df['LeafK'].mean() ,df['LeafMg'].mean() ,df['LeafB'].mean() ,df['YAN'].mean() ,df['TSS'].mean() ,df['pH'].mean() ,df['TA'].mean() ,df['raingrowing'].mean() ,df['tempgrowing'].mean() ,df['rainsepoct'].mean() ,df['tempsepoc'].mean() ,'Predicted value' ,df['trend'].mean() ,'Treatment' ,'Vineyard']


df3 = df2.iloc[:,1:16]
        


measurements = df3.drop(labels=["color"], axis=1).columns.tolist()

x_axis = st.sidebar.selectbox("X-Axis", measurements)
y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)



    malignant_df = breast_cancer_df[breast_cancer_df["color"] == "Original values"]
    benign_df = breast_cancer_df[breast_cancer_df["color"] == "Predicted value"]
    
    
        rain_temp = pd.read_csv("./rain_temp.csv")
