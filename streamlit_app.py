import streamlit as st
import pandas as pd
import numpy as np



@st.cache(allow_output_mutation=True)
def load_data():
    growmark_locs = pd.read_csv("")
    return growmark_locs


def main():

    logo = './img/cropped-hires_vineyard_nutrition_logo_color-270x270.png'
    st.sidebar.image(logo, width=200)

    st.sidebar.title('SCRI Yield/Quality Model')

    st.sidebar.markdown(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor 
        \n 
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor 
        """
    )

    st.sidebar.markdown(
        """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor 
        """
    )

    st.sidebar.text("")

    # body ===============================================================

    model_type = st.sidebar.radio('Model', ('Yield', 'Quality'), index=0)
    
    if model_type == "Yield":
        model_coeff = np.array([26.86, 22.89, 13.90])

    if model_type == "Quality":
        carb_seq = []       

    st.sidebar.header(f"Nutrient Metrics")
    nitrogen = st.sidebar.text_input("Nitrogen (units)", "0.5")
    phosphorus = st.sidebar.text_input("Phosphorus (units)", "0.5")
    potassium = st.sidebar.text_input("Potassium (units)", "0.5")
    
 
    button_click = st.sidebar.button('Run Model')

    if button_click:
        nutrient_inputs = np.array([float(nitrogen), float(phosphorus), float(potassium)])
        prediction = np.sum(model_coeff * nutrient_inputs)

        st.sidebar.text(f"Predicted Yield: {prediction}")
    



if __name__ == "__main__":

    main()
