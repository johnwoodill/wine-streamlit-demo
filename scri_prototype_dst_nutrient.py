import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from libs.libs import *
from RFNutrModel import *

# from gkey import *

st.set_page_config(layout="wide")

G_API_KEY = st.secrets["G_API_KEY"]

def get_rootstock(rootstock):
    root_101_14=0
    root_3309=0
    root_44_53=0
    root_OWNR=0
    root_RIPG=0
    root_SHWM=0

    if rootstock == '101_14':
        root_101_14=1
    if rootstock == '3309':
        root_3309=1
    if rootstock == '44_53':
        root_44_53=1
    if rootstock == 'OWNR':
        root_OWNR=1
    if rootstock == 'RIPG':
        root_RIPG=1
    if rootstock == 'SHWM':
        root_SHWM=1

    return root_101_14, root_3309, root_44_53, root_OWNR, root_RIPG, root_SHWM


def get_treatment(treatment):
    treat_1cls=0
    treat_2cls=0
    treat_NoThin=0    

    if treatment == '1 cluster left per shoot':
        treat_1cls=1
    if treatment == '2 cluster left per shoot':
        treat_2cls=1
    if treatment == 'No thinning':
        treat_NoThin=1
    
    return treat_1cls, treat_2cls, treat_NoThin    



def proc_model(hide_main=False):
    prism_dat = load_prism()

    
    lon_lat = gmap_geocode_coords(str(zip_code), G_API_KEY)
    if lon_lat is None:
        lon_lat = "Bad zip code"
    # st.write(f"Longitude/Latitude: {lon_lat}")
    
    # Get environmental data
    gridNumber = get_nearest_grid(prism_dat, lon_lat=lon_lat)
    env_data = prism_dat[prism_dat['gridNumber'] == gridNumber]
    env_data = env_data[env_data['date'] >= '2013-01-01'].sort_values('date').reset_index(drop=True)
    env_data = env_data.drop(columns=['gridNumber', 'lat', 'lon', 'tmin', 'tmax'])
    env_data = env_data[['date', 'tmean', 'ppt']]
    # st.write(env_data)

    Y_Tavg=env_data['tmean'].mean()
    Y_Pr=env_data['ppt'].mean()
    
    root_101_14, root_3309, root_44_53, root_OWNR, root_RIPG, root_SHWM = get_rootstock(rootstock)

    treat_1cls, treat_2cls, treat_NoThin = get_treatment(treatment)

    if hide_main == False:
        st.title("Model Inputs")
        st.text(f"Average Temp =  {Y_Tavg}")
        st.text(f"Average Precip = {Y_Pr}")

        st.text(f"Rootstock = {rootstock}")
        st.text(f"Treatment = {treatment}")
        st.text(f"Prev. Pruning Weight = {lag1_PWMRow}")
        
        st.text(f"Prev. Leaf Nitrogen = {lag1_LeafN}")
        
        st.text(f"Leaf Nitrogen = {LeafN}")
        st.text(f"Leaf Potassium = {LeafK}")
        st.text(f"Leaf Phosphorus = {LeafP}")

    gen_pred = gen_prediction( 
                LeafN,
                LeafP,
                LeafK,
                Y_Tavg,
                Y_Pr,
                lag1_LeafN,
                lag1_PWMRow,
                root_101_14,
                root_3309,
                root_44_53,
                root_OWNR,
                root_RIPG,
                root_SHWM,
                treat_1cls,
                treat_2cls,
                treat_NoThin)

    curr_results = pd.DataFrame({
        'pred_yield': gen_pred, 
        'rootstock': [rootstock], 
        'treatment': [treatment],
        'prev_PW': [lag1_PWMRow],
        'prev_N': [lag1_LeafN],
        'LeafN': [LeafN],
        'LeafK': [LeafK], 
        'LeafP': [LeafP],
        'tavg': [Y_Tavg], 
        'pr': [Y_Pr]})

    st.session_state.df = st.session_state.df.append(curr_results)
    st.session_state.df = st.session_state.df.drop_duplicates()

    return gen_pred




# def plot_ts(DF, tooltip = False, rangeSelector = False, width = 600, height = 400):
#   DF_Q = DF.select_dtypes(include = ['int64', 'float64'])
#   DF_T = DF.select_dtypes(include = ['datetime64[ns]'])
  
#   res = alt.Chart(DF).transform_fold(
#     DF_Q.columns.values, as_ = ['series', 'value']
#   ).mark_line().encode(
#     x = DF_T.columns.values[0] + ':T', y = 'value:Q', color = 'series:N'
#   ).properties(width = width, height = height)
  
#   selection = alt.selection_single(
#     fields = DF_T.columns.values.tolist(), nearest = True, 
#     on = 'mouseover', empty = 'none', clear = 'mouseout'
#   )
  
#   rule = alt.Chart(DF).mark_rule().encode(
#     x = DF_T.columns.values[0] + ':T',
#     opacity = alt.condition(selection, alt.value(0.3), alt.value(0))
#   ).add_selection(selection)
  
#   if tooltip == True:
#     points = res.mark_point().transform_filter(selection)
    
#     rule = alt.Chart(DF).mark_rule().encode(
#       x = DF_T.columns.values[0] + ':T',
#       opacity = alt.condition(selection, alt.value(0.3), alt.value(0)),
#       tooltip = np.append(DF_T.columns.values, DF_Q.columns.values).tolist()
#     ).add_selection(selection)
    
#     if rangeSelector == False:
#       return(res + points + rule)
  
#   if rangeSelector == True:
#     brush = alt.selection(type = 'interval', encodings = ['x'])
#     res = res.encode(
#       alt.X(DF_T.columns.values[0] + ':T', scale = alt.Scale(domain = brush))
#     )
    
#     points = res.mark_point().transform_filter(selection)
    
#     rango = alt.Chart(DF).mark_area().encode(
#       x = alt.X(DF_T.columns.values[0] + ':T', title = ''), 
#       y = alt.Y(DF_Q.columns.values[0] + ':Q', title = '', axis = alt.Axis(labels = False))
#     ).add_selection(brush).properties(width = width, height = (height * 0.1))
    
#     res = alt.vconcat(alt.layer(res, points, rule, points), rango)
  
#   return(res)


# !!!! UNCOMMENT WHEN PRISM IS AVAILABLE
# Load prism data into memory for faster processing

@st.cache(allow_output_mutation=True)
def load_prism():
    prism = pd.read_csv("data/prism_dat.csv")
    return prism



# with st.sidebar:
#     # logo = './img/cropped-hires_vineyard_nutrition_logo_color-270x270.png'
#     # st.image(logo, width=300)
#     st.write("Please choose among the options below to access different tabs of this app.")
#     # page_type = st.radio('Page', ('Home', 'Yield', 'Quality', 'Plots and analysis'), index=1)
#     page_type = st.radio('Page', ('Home', 'Yield', 'Plots and analysis'), index=0)

st.sidebar.image('./img/cropped-hires_vineyard_nutrition_logo_color-270x270.png', width=300)

page_type = st.sidebar.radio('', ('Main', 'Previous Results'), index=0)

st.sidebar.title("Vineyard Information")

variety = st.sidebar.selectbox("Wine-grape Variety",
    ('Pinot-Noir-1', 'Pinot-Noir-2'))

 # st.header(f"Vineyard Zipcode:")
zip_code = st.sidebar.text_input("Willamette Valley Zipcode", "97330")
# zip_code_click = st.sidebar.button('Get Environmental Data')

st.sidebar.title("Vineyard Management Decisions")

rootstock = st.sidebar.selectbox("Rootstock", 
    ('3309', 'RIPG', 'SHWM', 'OWNR', '101_14', '44_53'))

treatment = st.sidebar.selectbox("Thinning Practice",
    ('No thinning', '1 cluster left per shoot', '2 cluster left per shoot'))

lag1_PWMRow = st.sidebar.text_input("Previous Year's Pruning Weight (kg/m)", 0.50)
lag1_PWMRow = float(lag1_PWMRow)

st.sidebar.title("Nutrient Inputs")

lag1_LeafN = st.sidebar.text_input("Previous Year's Leaf Nitrogen (%)", 2.0)
lag1_LeafN = float(lag1_LeafN)

LeafN = st.sidebar.text_input("Expected Leaf Nitrogen (%)", 2.30)
LeafN = float(LeafN)

LeafK = st.sidebar.text_input("Expected Leaf Potassium (%)", 1.00)
LeafK = float(LeafK)

LeafP = st.sidebar.text_input("Expected Leaf Phosphorus (%)", 0.20)
LeafP = float(LeafP)

model_click = st.sidebar.button("Generate Model Prediction")

prev_results = pd.DataFrame()

# st.dataframe(prev_results)

if 'df' not in st.session_state:
    st.session_state.df = prev_results

def main():

    if page_type == "Main":


        st.title('Grape-Wine Nutrient Decision Support Tool')

        # st.markdown(
        #     """
        #     This app is part of the work of the HiRes Vineyard nutrition project, 
        #     \n 
        #     Please navigate this app by choosing the tab of interest on the sidebar of the screen 
        #     \n
        #     John, we´re obviously miles away from the perfect text here hehehee. But I guess the idea is 3 parragraphs:
        #         \n 
        #         1- Cordial welcome to the app and say what institutions and projects are behind it 
        #         \n 
        #         2- Very brief explanation of what this does: bascially take data from Oregon and make it useful to winemakers and people interested in wine in the PNW. 
        #         \n 
        #         3- Say what the sidebar is and that one can access the other tabs by choosing the right buttton. This will be self-evident from the app I think, but it never hurts to make it explicit.
        #     """
        # )

        st.markdown(
            """
            This decision support tool provides predicted yield (kg/m) using vineyard management decisions, 
            environmental data, and nutrient uptake values. 
            \n
            \n Note: these results are strictly for Pinot Noir in the Willamette Valley
            \n

        ------------------------------------------------------------------------------
            Instructions: 
                1. Select wine-grape variety (only Pinot-Noir is available)
                2. Input zip code to download environmental data 
                   temperature and precipitation
                
                3. Select vineyard management decisions for 
                   rootstock type, thinning practices, and previous year's 
                   pruning weight (kg/m)
                
                4. Input nutrients for previous year's nitrogen, and 
                   current year (expected) nitrogen, potassium, phosphorus
                
                5. Click "Generated Model Prediction"

            Note: each time you make a change on the left panel, click "Generate Model Prediction"

        ------------------------------------------------------------------------------

            """
        )
        
        # st.header(f"Vineyard Zipcode:")
        # zip_code = st.text_input("Zipcode", "97365")
        # zip_code_click = st.button('Get Environmental Data')

        # if zip_code_click:
        #     lon_lat = gmap_geocode_coords(str(zip_code), G_API_KEY)
        #     if lon_lat is None:
        #         lon_lat = "Bad zip code"
        #     st.write(f"Longitude/Latitude: {lon_lat}")
            
        #     # Get environmental data
        #     gridNumber = get_nearest_grid(prism_dat, lon_lat=lon_lat)
        #     env_data = prism_dat[prism_dat['gridNumber'] == gridNumber]
        #     env_data = env_data[env_data['date'] >= '2013-01-01'].sort_values('date').reset_index(drop=True)
        #     st.write(env_data)

        if model_click:
            gen_pred = proc_model()
            # Load prism data for search
            # prism_dat = load_prism()

            # st.title("Generating environmental data")
            # lon_lat = gmap_geocode_coords(str(zip_code), G_API_KEY)
            # if lon_lat is None:
            #     lon_lat = "Bad zip code"
            # st.write(f"Longitude/Latitude: {lon_lat}")
            
            # # Get environmental data
            # gridNumber = get_nearest_grid(prism_dat, lon_lat=lon_lat)
            # env_data = prism_dat[prism_dat['gridNumber'] == gridNumber]
            # env_data = env_data[env_data['date'] >= '2013-01-01'].sort_values('date').reset_index(drop=True)
            # env_data = env_data.drop(columns=['gridNumber', 'lat', 'lon', 'tmin', 'tmax'])
            # env_data = env_data[['date', 'tmean', 'ppt']]
            # st.write(env_data)

            # Y_Tavg=env_data['tmean'].mean()
            # Y_Pr=env_data['ppt'].mean()
            
            # root_101_14, root_3309, root_44_53, root_OWNR, root_RIPG, root_SHWM = get_rootstock(rootstock)

            # treat_1cls, treat_2cls, treat_NoThin = get_treatment(treatment)

            # st.title("Model Inputs")
            # st.text(f"Average Temp =  {Y_Tavg}")
            # st.text(f"Average Precip = {Y_Pr}")

            # st.text(f"Rootstock = {rootstock}")
            # st.text(f"Treatment = {treatment}")
            # st.text(f"Prev. Pruning Weight = {lag1_PWMRow}")
            
            # st.text(f"Prev. Leaf Nitrogen = {lag1_LeafN}")
            
            # st.text(f"Leaf Nitrogen = {LeafN}")
            # st.text(f"Leaf Potassium = {LeafK}")
            # st.text(f"Leaf Phosphorus = {LeafP}")

            # gen_pred = gen_prediction( 
            #             LeafN,
            #             LeafP,
            #             LeafK,
            #             Y_Tavg,
            #             Y_Pr,
            #             lag1_LeafN,
            #             lag1_PWMRow,
            #             root_101_14,
            #             root_3309,
            #             root_44_53,
            #             root_OWNR,
            #             root_RIPG,
            #             root_SHWM,
            #             treat_1cls,
            #             treat_2cls,
            #             treat_NoThin)

            # curr_results = pd.DataFrame({
            #     'pred_yield': gen_pred, 
            #     'rootstock': [rootstock], 
            #     'treatment': [treatment],
            #     'prev_PW': [lag1_PWMRow],
            #     'prev_N': [lag1_LeafN],
            #     'LeafN': [LeafN],
            #     'LeafK': [LeafK], 
            #     'LeafP': [LeafP],
            #     'tavg': [Y_Tavg], 
            #     'pr': [Y_Pr]})

            # st.session_state.df = st.session_state.df.append(curr_results)
            # st.session_state.df = st.session_state.df.drop_duplicates()

            st.title(f"Predicted Yield: {np.round(gen_pred, 2)} kg/m")

        

    if page_type == "Previous Results":
        if len(st.session_state.df) > 0:
        
            st.title(f"Previous Model Results")
            st.write(st.session_state.df)
            clear_previous_results = st.button("Clear Previous Model Results")

            if clear_previous_results:
                st.session_state.df = pd.DataFrame()

            if model_click:
                gen_pred = proc_model(hide_main=True)
                

            pdat = st.session_state.df
            pdat['pred_yield'] = pdat['pred_yield'].astype(float)
            pdat['LeafN'] = pdat['LeafN'].astype(float)
            pdat['LeafK'] = pdat['LeafK'].astype(float)
            pdat['prev_PW'] = pdat['prev_PW'].astype(float)
            pdat['prev_N'] = pdat['prev_N'].astype(float)
            pdat['label'] = pdat['treatment'].astype(str) + "-" + pdat['rootstock'].astype(str)

            fig, ax = plt.subplots()
            sns.scatterplot(pdat['LeafN'], pdat['pred_yield'], hue=pdat['label'], ax=ax)
            ax.set_xlim(0,5)
            ax.set_xticks(np.arange(0, 5.5, 0.5))
            ax.legend(title='', loc='best')
            ax.set_ylim(0,5)
            ax.set_yticks(np.arange(0, 5.5, 0.5))
            plt.xlabel('Nitrogen (%)')
            plt.ylabel('Predicted Yield (kg/m)')
            plt.title("Nitrogen and Predicted Yield \n by Thinning and Rootstock")
            st.pyplot(fig)


            fig, ax = plt.subplots()
            sns.scatterplot(pdat['LeafK'], pdat['pred_yield'], hue=pdat['label'], ax=ax)
            ax.set_xlim(0,5)
            ax.set_xticks(np.arange(0, 5.5, 0.5))
            ax.legend(title='', loc='best')
            ax.set_ylim(0,5)
            ax.set_yticks(np.arange(0, 5.5, 0.5))
            plt.xlabel('Potassium (%)')
            plt.ylabel('Predicted Yield (kg/m)')
            plt.title("Potassium and Predicted Yield \n by Thinning and Rootstock")
            st.pyplot(fig)


            fig, ax = plt.subplots()
            sns.scatterplot(pdat['prev_PW'], pdat['pred_yield'], hue=pdat['label'], ax=ax)
            ax.set_xlim(0,5)
            ax.set_xticks(np.arange(0, 5.5, 0.5))
            ax.legend(title='', loc='best')
            ax.set_ylim(0,5)
            ax.set_yticks(np.arange(0, 5.5, 0.5))
            plt.xlabel("Previous Year's Pruning Weights")
            plt.ylabel('Predicted Yield (kg/m)')
            plt.title("Previous Pruning Weights and Predicted Yield \n by Thinning and Rootstock")
            st.pyplot(fig)


            fig, ax = plt.subplots()
            sns.scatterplot(pdat['prev_N'], pdat['pred_yield'], hue=pdat['label'], ax=ax)
            ax.set_xlim(0,5)
            ax.set_xticks(np.arange(0, 5.5, 0.5))
            ax.legend(title='', loc='best')
            ax.set_ylim(0,5)
            ax.set_yticks(np.arange(0, 5.5, 0.5))
            plt.xlabel("Previous Year's Nitrogen")
            plt.ylabel('Predicted Yield (kg/m)')
            plt.title("Previous Nitrogen and Predicted Yield \n by Thinning and Rootstock")
            st.pyplot(fig)

        else:
            st.write("No previous results to display")


    #         # if 'prediction_yield' not in st.session_state:
    #         st.session_state['prediction_yield'] = prediction
            
    #         # st.write(st.session_state)

    #         st.text(f"Predicted Yield: {np.round(prediction,2)} per kg/m")
            

            
    # if page_type == "Quality":
    #     st.title('Quality prediction model')
    #     model_coeff = np.array([26.86, 22.89, 13.90,1.82,-0.04])

    #     st.markdown(
    #         """
    #         Below, you may input the values of different nutrients found in the leaves of a Pinot Noir winegrape in Oregon and you will see a predicted quality level of the wine. 
    #         \n 
    #         At the moment, this is the same as the yield model, but once we have something we can just replace the values in the boxes.
    #         """
    #     )
    #     # body ===============================================================

    

    #     st.header(f"Nutrient Metrics")
    #     nitrogen = st.text_input("Leaf Nitrogen (%)", "2.211")
    #     phosphorus = st.text_input("Phosphorus (%)", "0.182")
    #     potassium = st.text_input("Potassium (%)", "1.036")
    #     magnesium = st.text_input("Magnesium (%)", "0.329")
    #     boron = st.text_input("Boron (%)", "30.730")

    #     button_click = st.button('Run Model')

    #     if button_click:
    #         nutrient_inputs = np.array([float(nitrogen), float(phosphorus), float(potassium), float(magnesium), float(boron)])
    #         prediction = np.sum(model_coeff * nutrient_inputs)

    #         st.text(f"Predicted Quality: {prediction} (Wine Spectator Rating)")
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
    #     # page ===============================================================
        
    # if page_type == "Plots and analysis":
    #     st.title('Plots and analysis')


    #     # st.markdown(
    #     #     """
    #     #     This page displays plots that you can manipulate. To add the predicted yield value in the plots, please insert the nutrient values in the  "Yield" page.
    #     #     """
    #     # )
    #     # body ===============================================================
        
    #     rain_temp = pd.read_csv("./rain_temp.csv")
        
    #     df = pd.read_csv("./patty_reg.csv")
       
       
    #     # st.write(df['yield_kg_m'].mean())
        
    #     df2 = pd.read_csv("./patty_reg.csv")
        
    #     if 'prediction_yield' not in st.session_state:
    #         st.session_state['prediction_yield'] = 0
        
    #     prediccion= st.session_state['prediction_yield']
        
        
    #     df2.loc[len(df2.index)] = [999,prediccion,df['LeafN'].mean() ,df['LeafP'].mean() ,df['LeafK'].mean() ,df['LeafMg'].mean() ,df['LeafB'].mean() ,df['YAN'].mean() ,df['TSS'].mean() ,df['pH'].mean() ,df['TA'].mean() ,df['raingrowing'].mean() ,df['tempgrowing'].mean() ,df['rainsepoct'].mean() ,df['tempsepoc'].mean(), 'Predicted value' ,df['trend'].mean() ,'Treatment' ,'Vineyard']
        
        
        
    #     df3 = df2.iloc[:,1:16]
        
    #     #st.dataframe(df3)
        
        
    #     measurements = df3.drop(labels=["color"], axis=1).columns.tolist()
        
    #     st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Yield and Nutrients :")

    #     x_axis = st.sidebar.selectbox("X-Axis", measurements)
    #     y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)

    #     if x_axis and y_axis:
    #         scatter_fig = plt.figure(figsize=(4,2))

    #         scatter_ax = scatter_fig.add_subplot(111)
        

    #         malignant_df = df3[df3["color"] == "Original values"]
    #         benign_df = df3[df3["color"] == "Predicted value"]

    #         malignant_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label="Original values")
    #         benign_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
    #                            title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Predicted value");
            

                    
    #     st.sidebar.markdown("### Histogram: Explore Distribution of yield and nutrients: ")

    #     hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=measurements, default=["yield_kg_m", "LeafN"])
    #     bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)

    #     if hist_axis:
    #         hist_fig = plt.figure(figsize=(4,2))
    #         hist_ax = hist_fig.add_subplot(111)
    #         sub_breast_cancer_df = df3[hist_axis]
    #         sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
    #     else:
    #         hist_fig = plt.figure(figsize=(4,2))
    #         hist_ax = hist_fig.add_subplot(111)
    #         sub_breast_cancer_df = df3[["mean radius", "mean texture"]]
    #         sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
            
            
    #     container1 = st.container()
    #     col1, col2 = st.columns(2)

    #     with container1:
    #         with col1:
    #             scatter_fig            
    #         with col2:
    #             hist_fig
                
    #     fechas = pd.DatetimeIndex(rain_temp.Date)
    #     ts = pd.Series(rain_temp.rains.values, index = fechas)
    #     ts_rain = ts.to_frame(name = "Precip").reset_index()
    #     ts_plotrain = plot_ts(ts_rain, True, True)

    #     #rain=st.altair_chart(ts_plotrain, use_container_width=True) 
        
    #     ts2 = pd.Series(rain_temp.temps.values, index = fechas)
    #     ts_df = ts2.to_frame(name = "Temperature").reset_index()
    #     ts_plot = plot_ts(ts_df, True, True)
    #     #temp=st.altair_chart(ts_plot, use_container_width=True)        
        
    #     container2 = st.container()
    #     col3, col4 = st.columns(2)

    #     with container2:
    #         with col3:
    #             ts_plotrain            
    #         with col4:
    #             ts_plot        
       
        
        

if __name__ == "__main__":

    main()
