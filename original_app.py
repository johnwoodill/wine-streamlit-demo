# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:14:51 2022

@author: alejo
"""

import streamlit as st
import pandas as pd
import numpy as np



@st.cache(allow_output_mutation=True)
def load_data():
    growmark_locs = pd.read_csv("")
    return growmark_locs



with st.sidebar:
    logo = './img/cropped-hires_vineyard_nutrition_logo_color-270x270.png'
    st.image(logo, width=300)
    st.write("So I  think in this part we should write some welcome bla bla bla, and then have three options of pages to go to: yield, quality, info about Oregon (or something like that)")
    
    page_app = st.radio('Page', ('Home', 'Yield', 'Quality', 'Plots or something'), index=0)

def main():
    
    


    st.title('SCRI Yield/Quality Model')

    st.markdown(
        """
        So we would have 4 "pages" (tabs?). 
        \n 
        This first tab would explain what this tab is and would give a little bit more information about what this app is and how it is intended to be used. Nothing to complex nor long, 3 parragrpahs at most. Just some text to let anyone be clear about the intentions and reach of this app 
        """
    )



    st.text("")

    # body ===============================================================

    model_type = st.radio('Model', ('Yield', 'Quality'), index=0)
    
    if model_type == "Yield":
        model_coeff = np.array([26.86, 22.89, 13.90,1.82,-0.04])

    if model_type == "Quality":
        carb_seq = []       

    st.header(f"Nutrient Metrics")
    nitrogen = st.text_input("Leaf Nitrogen (%)", "2.211")
    phosphorus = st.text_input("Phosphorus (%)", "0.182")
    potassium = st.text_input("Potassium (%)", "1.036")
    magnesium = st.text_input("Magnesium (%)", "0.329")
    boron = st.text_input("Boron (%)", "30.730")

    button_click = st.button('Run Model')

    if button_click:
        nutrient_inputs = np.array([float(nitrogen), float(phosphorus), float(potassium), float(magnesium), float(boron)])
        prediction = np.sum(model_coeff * nutrient_inputs)

        st.text(f"Predicted Yield: {prediction} per kg/m")
    



if __name__ == "__main__":

    main()
