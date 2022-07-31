# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 09:27:09 2022

@author: alejo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


st.set_page_config(page_title = "Yield prediction")





st.title('Wine yield prediction model')
model_coeff = np.array([26.86, 22.89, 13.90,1.82,-0.04,-0.11,-5.39,13.35,8.28])

st.markdown(
            """
            Below, you may input the values of different nutrients found in the leaves of a Pinot Noir winegrape in Oregon and you will see a predicted yield level in kilograms by meter. 
            \n 
            Again, we gotta write this better. But I think this is the strcuture: ery brief and self-exlanatory
            """
        )
        # body ===============================================================

    

st.header(f"Nutrient Metrics")
nitrogen = st.text_input("Leaf Nitrogen (%)", "2.211")
phosphorus = st.text_input("Phosphorus (%)", "0.182")
potassium = st.text_input("Potassium (%)", "1.036")
magnesium = st.text_input("Magnesium (%)", "0.329")
boron = st.text_input("Boron (%)", "30.730")
YAN = st.text_input("YAN", "150.718")
TSS = st.text_input("TSS", "23.335")
pH = st.text_input("pH", "3.362")
TA = st.text_input("TA", "6.563")
        
raings= 4.41*6.418
tempgs=34.38*16.445
rainsepoct=-2.35*6.189
tempsepoct=2.69*14.813
        

        


button_click = st.button('Run Model')
if button_click:
    nutrient_inputs = np.array([float(nitrogen), float(phosphorus), float(potassium), float(magnesium), float(boron), float(YAN), float(TSS), float(pH), float(TA)])
            
    st.session_state
                      
            
    prediction = np.sum(model_coeff * nutrient_inputs)+1
            
    if 'prediction_yield' not in st.session_state:
        st.session_state['prediction_yield'] = prediction
            
    st.write(st.session_state)

    st.text(f"Predicted Yield: {prediction} per kg/m")
            

            
            