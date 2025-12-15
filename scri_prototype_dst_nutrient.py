import streamlit as st
import pandas as pd
import numpy as np

from libs.libs import *
from RFNutrModel import *
from gkey import *

st.set_page_config(layout="wide")


# -----------------------------
# Utility functions
# -----------------------------
def get_rootstock(rootstock):
    return (
        int(rootstock == '101_14'),
        int(rootstock == '3309'),
        int(rootstock == '44_53'),
        int(rootstock == 'OWNR'),
        int(rootstock == 'RIPG'),
        int(rootstock == 'SHWM'),
    )


def get_treatment(treatment):
    return (
        int(treatment == '1 cluster left per shoot'),
        int(treatment == '2 cluster left per shoot'),
        int(treatment == 'No thinning'),
    )


# -----------------------------
# Cached data
# -----------------------------
@st.cache_data
def load_prism():
    return pd.read_csv("data/prism_dat.csv")


# -----------------------------
# Core model runner
# -----------------------------
def proc_model(hide_main=False):
    prism_dat = load_prism()

    lon_lat = gmap_geocode_coords(str(zip_code), G_API_KEY)
    if lon_lat is None:
        st.error("Invalid zip code")
        return None

    grid_number = get_nearest_grid(prism_dat, lon_lat=lon_lat)

    env_data = (
        prism_dat.loc[prism_dat['gridNumber'] == grid_number]
        .loc[lambda x: x['date'] >= '2013-01-01']
        .sort_values('date')
        .reset_index(drop=True)
        .drop(columns=['gridNumber', 'lat', 'lon', 'tmin', 'tmax'])
        [['date', 'tmean', 'ppt']]
    )

    Y_Tavg = env_data['tmean'].mean()
    Y_Pr = env_data['ppt'].mean()

    root_vals = get_rootstock(rootstock)
    treat_vals = get_treatment(treatment)

    if not hide_main:
        st.title("Model Inputs")
        st.write(f"Average Temp: {Y_Tavg:.2f}")
        st.write(f"Average Precip: {Y_Pr:.2f}")
        st.write(f"Rootstock: {rootstock}")
        st.write(f"Treatment: {treatment}")

    gen_pred = gen_prediction(
        LeafN,
        LeafP,
        LeafK,
        Y_Tavg,
        Y_Pr,
        lag1_LeafN,
        lag1_PWMRow,
        *root_vals,
        *treat_vals,
    )

    curr_results = pd.DataFrame(
        [{
            'pred_yield': gen_pred,
            'rootstock': rootstock,
            'treatment': treatment,
            'prev_PW': lag1_PWMRow,
            'prev_N': lag1_LeafN,
            'LeafN': LeafN,
            'LeafK': LeafK,
            'LeafP': LeafP,
            'tavg': Y_Tavg,
            'pr': Y_Pr,
        }]
    )

    st.session_state.df = (
        pd.concat([st.session_state.df, curr_results], ignore_index=True)
        .drop_duplicates()
    )

    return gen_pred


# -----------------------------
# Sidebar UI
# -----------------------------
st.sidebar.image(
    './img/cropped-hires_vineyard_nutrition_logo_color-270x270.png',
    width=300
)

st.sidebar.title("Vineyard Information")

variety = st.sidebar.selectbox(
    "Wine-grape Variety",
    ('Pinot-Noir-1', 'Pinot-Noir-2')
)

zip_code = st.sidebar.text_input("Willamette Valley Zipcode", "97330")

st.sidebar.title("Vineyard Management Decisions")

rootstock = st.sidebar.selectbox(
    "Rootstock",
    ('3309', 'RIPG', 'SHWM', 'OWNR', '101_14', '44_53')
)

treatment = st.sidebar.selectbox(
    "Thinning Practice",
    ('No thinning', '1 cluster left per shoot', '2 cluster left per shoot')
)

lag1_PWMRow = float(
    st.sidebar.text_input("Previous Year's Pruning Weight (kg/m)", 0.50)
)

st.sidebar.title("Nutrient Inputs")

lag1_LeafN = float(
    st.sidebar.text_input("Previous Year's Leaf Nitrogen (%)", 2.0)
)

LeafN = float(
    st.sidebar.text_input("Expected Leaf Nitrogen (%)", 2.30)
)

LeafK = float(
    st.sidebar.text_input("Expected Leaf Potassium (%)", 1.00)
)

LeafP = float(
    st.sidebar.text_input("Expected Leaf Phosphorus (%)", 0.20)
)

model_click = st.sidebar.button("Predict Yield")


# -----------------------------
# App state
# -----------------------------
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()


# -----------------------------
# Main
# -----------------------------
def main():
    st.title('Wine-grape Nutrient Decision Support Tool')

    st.markdown(
        """
        This decision support tool uses a Yield-Nutrient model to predict 
        end-of-season yield (kg/m). The Yield-Nutrient model was developed 
        using a Random Forest Regressor with vineyard management decisions, 
        environmental data, and nutrient uptake values. The data used were 
        Patty Skinkis's State Wide Crop Load data for the Willamette Valley
        in Oregon from 2013-2021. The model was able to capture 60% of
        variation in the data (R-squared = 60%) using a Leave-One-Out
        Cross-validation technique.
        \n
        \n Note: these results are strictly for Pinot Noir in the Willamette Valley
                

    ------------------------------------------------------------------------------
        Instructions: 
            1. Select wine-grape variety (only Pinot-Noir is available)
            2. Input zip code to download environmental data 
               temperature and precipitation
            
            3. Select vineyard management decisions for 
               rootstock type, thinning practices, and previous year's 
               pruning weight (kg/m)
            
            4. Input nutrients for previous year's nitrogen, and 
               current year (expected) nitrogen, potassium, and 
               phosphorus
            
            5. Click "Predict Yield"

    ------------------------------------------------------------------------------

        """
    )

    if model_click:
        gen_pred = proc_model()
        if gen_pred is not None:
            st.header(f"Predicted Yield: {gen_pred:.2f} kg/m")


if __name__ == "__main__":
    main()
