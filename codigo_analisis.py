# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 19:33:02 2022

@author: alejo
"""


    df2.loc[len(df2.index)] = [999,3,df['LeafN'].mean() ,df['LeafP'].mean() ,df['LeafK'].mean() ,df['LeafMg'].mean() ,df['LeafB'].mean() ,df['YAN'].mean() ,df['TSS'].mean() ,df['pH'].mean() ,df['TA'].mean() ,df['raingrowing'].mean() ,df['tempgrowing'].mean() ,df['rainsepoct'].mean() ,df['tempsepoc'].mean() ,'Predicted value' ,df['trend'].mean() ,'Treatment' ,'Vineyard']
        
        # st.dataframe(df2)
        
    df3 = df2.iloc[:,1:16]


    measurements = df3.drop(labels=["color"], axis=1).columns.tolist()

    x_axis = st.sidebar.selectbox("X-Axis", measurements)
    y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)

    if x_axis and y_axis:
        scatter_fig = plt.figure(figsize=(6,4))

    scatter_ax = scatter_fig.add_subplot(111)
    

    malignant_df = df3[df3["color"] == "Original values"]
    benign_df = df3[df3["color"] == "Predicted value"]

    malignant_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label="Original values")
    benign_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
                           title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Predicted value");
         
        
    st.markdown(
            """
            Here, we will first show how yield is distributed. 
            \n 
            It doesn´t necessarily have to be a histogram, but I think people like them.
            """
        )
        
    rain_temp = pd.read_csv("./rain_temp.csv")
                
        
    fig, ax = plt.subplots()
    df.hist(    bins=20,  column="yield_kg_m",  grid=False,  figsize=(8, 8),  color="#86bf91",  zorder=2,  rwidth=0.9,  ax=ax,)
    st.write(fig)
 
    st.markdown(
            """
            Then, we can show some plots that show how nutrients and yield behave. I was thinking that it might be cool to add the possibilitiy of chosing what to plot, instead of always having the same plots.
            """
        )
        
        
    st.write(st.session_state['prediction_yield'])
        
        
        
               
    fig2, ax = plt.subplots(figsize=(8, 8), dpi = 200)

    def theme_minimal(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    sns.scatterplot(x = "yield_kg_m",
              y = "LeafN",
              hue= "Treatment",
              data = df,
              ax = ax)
    theme_minimal(ax)
    st.write(fig2)
        
        
    plot1= alt.Chart(df).mark_circle().encode(
        alt.X('LeafP', scale=alt.Scale(zero=False)),
        alt.Y('LeafN', scale=alt.Scale(zero=False, padding=1)),
        color='Treatment',
        size='yield_kg_m',
        tooltip=['LeafN:N','LeafP:N','yield_kg_m:N'])
    st.altair_chart(plot1, use_container_width=True)
        
    plot2= alt.Chart(df).mark_circle().encode(
        alt.X('LeafK', scale=alt.Scale(zero=False)),
        alt.Y('LeafN', scale=alt.Scale(zero=False, padding=1)),
        color='Treatment',
        size='yield_kg_m',
        tooltip=['LeafN:N','LeafK:N','yield_kg_m:N'])
    st.altair_chart(plot2, use_container_width=True)
        
    plot3= alt.Chart(df).mark_circle().encode(
        alt.X('yield_kg_m', scale=alt.Scale(zero=False)),
        alt.Y('LeafN', scale=alt.Scale(zero=False, padding=1)),
        color='LeafK',
        size='Treatment',
        tooltip=['LeafN:N','LeafK:N','yield_kg_m:N','Treatment:N'])
    st.altair_chart(plot3, use_container_width=True)
        
        
    fig3, ax = plt.subplots(figsize=(8, 8), dpi = 200)

    def theme_minimal(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    sns.scatterplot(x = "yield_kg_m",
              y = "LeafP",
              hue= "Treatment",
              data = df,
              ax = ax)
    theme_minimal(ax)
    st.write(fig3)
        
        
    plot4= alt.Chart(df).mark_circle().encode(
        alt.X('yield_kg_m', scale=alt.Scale(zero=False)),
        alt.Y('LeafP', scale=alt.Scale(zero=False, padding=1)),
        color='LeafK',
        size='Treatment',
        tooltip=['LeafN:N','LeafK:N','yield_kg_m:N','Treatment:N'])
    st.altair_chart(plot4, use_container_width=True)
        
    fechas = pd.DatetimeIndex(rain_temp.Date)
    ts = pd.Series(rain_temp.rains.values, index = fechas)

        # fig4, ax = plt.subplots(figsize=(8, 8), dpi = 200)
        # ts.plot()
        # plt.ylabel('Rainfall in mm')
        # plt.xlabel('Year')
        # st.pyplot(fig4)
        
    ts_df = ts.to_frame(name = "Rain").reset_index()
    ts_plot = plot_ts(ts_df, True, True)
    st.altair_chart(ts_plot, use_container_width=True)
  

    fig5, ax = plt.subplots(figsize=(8, 8), dpi = 200)
    ts2 = pd.Series(rain_temp.temps.values, index = fechas)
    ts2.plot()
    plt.ylabel('Temperature, celsius')
    plt.xlabel('Year')
    st.pyplot(fig5)
        
      
        
        
        
    ts_df = ts2.to_frame(name = "Temperature").reset_index()
    ts_plot = plot_ts(ts_df, True, True)
    st.altair_chart(ts_plot, use_container_width=True)


    



    

