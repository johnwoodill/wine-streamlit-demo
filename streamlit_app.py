import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from libs.libs import *

from gkey import *

st.set_page_config(layout="wide")


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


# !!!! UNCOMMENT WHEN PRISM IS AVAILABLE
# Load prism data into memory for faster processing

@st.cache(allow_output_mutation=True)
def load_prism():
    prism = pd.read_csv("data/prism_dat.csv")
    return prism



with st.sidebar:
    logo = './img/cropped-hires_vineyard_nutrition_logo_color-270x270.png'
    st.image(logo, width=300)
    st.write("Please choose among the options below to access different tabs of this app.")
    # page_type = st.radio('Page', ('Home', 'Yield', 'Quality', 'Plots and analysis'), index=1)
    page_type = st.radio('Page', ('Home', 'Yield', 'Plots and analysis'), index=0)

        

def main():
    
    if page_type == "Home":
        st.title('Welcome to the Yield Decision Support Tool')

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
            We gather environmental data based on the zip code of the vineyard. Please enter the zip code
            to get the closests environmental data to your location. 
            \n
            (Note: this information is only used for query Google Maps)
            """
        )
        
        # Load prism data for search
        prism_dat = load_prism()

        st.header(f"Vineyard Zipcode:")
        zip_code = st.text_input("Zipcode", "97365")
        zip_code_click = st.button('Get Environmental Data')

        if zip_code_click:
            lon_lat = gmap_geocode_coords(str(zip_code), G_API_KEY)
            if lon_lat is None:
                lon_lat = "Bad zip code"
            st.write(f"Longitude/Latitude: {lon_lat}")
            
            # Get environmental data
            gridNumber = get_nearest_grid(prism_dat, lon_lat=lon_lat)
            env_data = prism_dat[prism_dat['gridNumber'] == gridNumber]
            st.write(env_data)


        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
               
    if page_type == "Yield":
        st.title('Wine yield prediction model')
        model_coeff = np.array([0.2685, 0.2289, 0.1390,0.0181,-0.0003,-0.0011,-0.0539,0.1335,0.0828])

        st.markdown(
            """
            Below, you may input the values of different nutrients found in the leaves of a Pinot Noir winegrape in Oregon \n 
            and you will see a predicted yield level in kilograms by meter. 
            """
        )
        # body ===============================================================

    

        st.header(f"Nutrient Metrics")
        nitrogen = st.text_input("Leaf Nitrogen (%)", "2.211")
        # phosphorus = st.text_input("Phosphorus (%)", "0.182")
        phosphorus = 0.182
        potassium = st.text_input("Potassium (%)", "1.036")
        # magnesium = st.text_input("Magnesium (%)", "0.329")
        magnesium = 0.329
        # boron = st.text_input("Boron (%)", "30.730")
        boron = 30.730
        # YAN = st.text_input("YAN", "150.718")
        YAN = 150.718
        # TSS = st.text_input("TSS", "23.335")
        TSS = 23.335
        # pH = st.text_input("pH", "3.362")
        pH = 3.362
        # TA = st.text_input("TA", "6.563")
        TA = 6.563
        
        raings= 0.0441*6.418
        tempgs=0.3437*16.445
        rainsepoct=-0.0235*6.189
        tempsepoct=0.0268*14.813
        
        rest= 0.1910-0.3343+0.7259-0.1304+0.3537+2.225+1.1616+0.7331+0.9619+0.3612+1.0714+0.7481-0.8848-0.3079-0.2204+0.06877-0.1916-0.3869-0.2185+0.2549-0.3233-0.3748+0.2428-0.1795-0.1512+raings+tempgs+rainsepoct+tempsepoct-6.9526
        


        button_click = st.button('Run Model')

        if button_click:
            nutrient_inputs = np.array([float(nitrogen), float(phosphorus), float(potassium), float(magnesium), float(boron), float(YAN), float(TSS), float(pH), float(TA)])
            
            # st.session_state
            
            ## Please remember to erase the number 2, I'm just doi
                      
            
            prediction = np.sum(model_coeff * nutrient_inputs)+raings+tempgs+rainsepoct+tempsepoct-6.9526 + 2
            
            # if 'prediction_yield' not in st.session_state:
            st.session_state['prediction_yield'] = prediction
            
            # st.write(st.session_state)

            st.text(f"Predicted Yield: {np.round(prediction,2)} per kg/m")
            

            
           
    
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
            
    if page_type == "Quality":
        st.title('Quality prediction model')
        model_coeff = np.array([26.86, 22.89, 13.90,1.82,-0.04])

        st.markdown(
            """
            Below, you may input the values of different nutrients found in the leaves of a Pinot Noir winegrape in Oregon and you will see a predicted quality level of the wine. 
            \n 
            At the moment, this is the same as the yield model, but once we have something we can just replace the values in the boxes.
            """
        )
        # body ===============================================================

    

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

            st.text(f"Predicted Quality: {prediction} (Wine Spectator Rating)")
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        # page ===============================================================
        
    if page_type == "Plots and analysis":
        st.title('Plots and analysis')


        # st.markdown(
        #     """
        #     This page displays plots that you can manipulate. To add the predicted yield value in the plots, please insert the nutrient values in the  "Yield" page.
        #     """
        # )
        # body ===============================================================
        
        rain_temp = pd.read_csv("./rain_temp.csv")
        
        df = pd.read_csv("./patty_reg.csv")
       
       
        # st.write(df['yield_kg_m'].mean())
        
        df2 = pd.read_csv("./patty_reg.csv")
        
        if 'prediction_yield' not in st.session_state:
            st.session_state['prediction_yield'] = 0
        
        prediccion= st.session_state['prediction_yield']
        
        
        df2.loc[len(df2.index)] = [999,prediccion,df['LeafN'].mean() ,df['LeafP'].mean() ,df['LeafK'].mean() ,df['LeafMg'].mean() ,df['LeafB'].mean() ,df['YAN'].mean() ,df['TSS'].mean() ,df['pH'].mean() ,df['TA'].mean() ,df['raingrowing'].mean() ,df['tempgrowing'].mean() ,df['rainsepoct'].mean() ,df['tempsepoc'].mean(), 'Predicted value' ,df['trend'].mean() ,'Treatment' ,'Vineyard']
        
        
        
        df3 = df2.iloc[:,1:16]
        
        #st.dataframe(df3)
        
        
        measurements = df3.drop(labels=["color"], axis=1).columns.tolist()
        
        st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Yield and Nutrients :")

        x_axis = st.sidebar.selectbox("X-Axis", measurements)
        y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)

        if x_axis and y_axis:
            scatter_fig = plt.figure(figsize=(4,2))

            scatter_ax = scatter_fig.add_subplot(111)
        

            malignant_df = df3[df3["color"] == "Original values"]
            benign_df = df3[df3["color"] == "Predicted value"]

            malignant_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label="Original values")
            benign_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
                               title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Predicted value");
            

                    
        st.sidebar.markdown("### Histogram: Explore Distribution of yield and nutrients: ")

        hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=measurements, default=["yield_kg_m", "LeafN"])
        bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)

        if hist_axis:
            hist_fig = plt.figure(figsize=(4,2))
            hist_ax = hist_fig.add_subplot(111)
            sub_breast_cancer_df = df3[hist_axis]
            sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
        else:
            hist_fig = plt.figure(figsize=(4,2))
            hist_ax = hist_fig.add_subplot(111)
            sub_breast_cancer_df = df3[["mean radius", "mean texture"]]
            sub_breast_cancer_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
            
            
        container1 = st.container()
        col1, col2 = st.columns(2)

        with container1:
            with col1:
                scatter_fig            
            with col2:
                hist_fig
                
        fechas = pd.DatetimeIndex(rain_temp.Date)
        ts = pd.Series(rain_temp.rains.values, index = fechas)
        ts_rain = ts.to_frame(name = "Precip").reset_index()
        ts_plotrain = plot_ts(ts_rain, True, True)

        #rain=st.altair_chart(ts_plotrain, use_container_width=True) 
        
        ts2 = pd.Series(rain_temp.temps.values, index = fechas)
        ts_df = ts2.to_frame(name = "Temperature").reset_index()
        ts_plot = plot_ts(ts_df, True, True)
        #temp=st.altair_chart(ts_plot, use_container_width=True)        
        
        container2 = st.container()
        col3, col4 = st.columns(2)

        with container2:
            with col3:
                ts_plotrain            
            with col4:
                ts_plot        
       
        
        

if __name__ == "__main__":

    main()
