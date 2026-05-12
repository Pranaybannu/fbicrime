##streamli app
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
# Set the title of the app
st.set_page_config(page_title="FBI CRIME FORECASTING ", layout='wide')
st.title("FBI MONTHLY CRIME FORECASTING APP")
st.markdown("""
Choose the following forecasting models
            1. Overall monthly crime forecasting
            2. Negihbourhood Specific crime forecasting
            3. Crime type specific forecasting""")

##MODEL PATHS
stage1_model_path = r"C:\Users\prana\Downloads\fbicrime\stage1sarimax.pkl"

##neighbourhood specific model paths
main_neighbour = r"C:\Users\prana\Downloads\fbicrime\neighbourhood models"

##crime type specific model paths
main_crime= r"C:\Users\prana\Downloads\fbicrime\crimetype models"

##loading stage1 model
@st.cache_data
def load_stage1_model():
    return joblib.load(stage1_model_path)

##loading neighbourhood models
@st.cache_data
def load_neighbour_model(neighbourhood):
    model_path = f"{main_neighbour}/stage2{neighbourhood}sarimax.pkl"
    return joblib.load(model_path)

##laoding neighboruhoods
def load_neighbours(folder_path=Path(main_neighbour)):
    if not folder_path.exists():
        st.error("The specified folder path does not exist.")
        return []
    else:
        neighbourhoods = [] 
        for c in folder_path.glob("*.pkl"):
            name = c.stem
            if name.startswith("stage2") and name.endswith("sarimax"):
                name = name[len("stage2"):]
                name=name[:-len("sarimax")]
                name = name.strip("'").strip()   # Remove any single quotes
                neighbourhoods.append(name)
        return neighbourhoods


        
##loading crime type models
@st.cache_data
def load_crime_model(crime_type):
    model_path = f"{main_crime}/stage2'{crime_type}'sarimax.pkl"
    return joblib.load(model_path)

###laoding crime types
def load_crime_types(folder_path=Path(main_crime)):
    if not folder_path.exists():
        st.error("The specified folder path doesn't exist.")
        return []
    else:
        crime_types = []    
        for c in folder_path.glob("*.pkl"):
            name = c.stem
            if name.startswith("stage2") and name.endswith("sarimax"):
                name = name[len("stage2"):-len("sarimax")]
                name = name.strip("'").strip()  # Remove any single quotes
                crime_types.append(name)
        return crime_types
    

##ploting forecasted values
def plot_forecast(data,title):
    df = pd.DataFrame({'Date':data.index,
                       "Forecast":data.values})
    
    st.dataframe(df,width=True)
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'],df['Forecast'],marker='o',lw=2)
    plt.title(title,weight='bold',fontsize=16)
    plt.xlabel("Date",weight='bold',fontsize=12)
    plt.ylabel("Forecast",weight='bold',fontsize=12)
    plt.show()
    st.pyplot(plt)  


###FORECASTING DATA
st.sidebar.header("select any of the following forecasting models")
options = st.sidebar.radio(
    "CHOOSE THE FORECASTING MODEL",
    [
        "Overall monthly crime forecasting",
        "Neighbourhood Specific crime forecasting",
        "Crime type specific forecasting"
    ])

months = st.sidebar.slider("select the numerb of months to forecast",
                           min_value=1,
                           max_value=24,
                           value=12)

##overall monthly crime forecasting
if options == "Overall monthly crime forecasting":
    st.subheader("overall monthly crime forecasting")
    try:
        stage1_model=load_stage1_model()
        forecast = stage1_model.forecast(steps=months)
        plot_forecast(data=forecast,title=f"OVERALL MONTHLY CRIME FORECASTING FOR {months}")
    except Exception as e:
        st.error(f"An error occurred while loading the model or forecasting: {e}")


##neighbourhood specific crime forecasting
elif options == "Neighbourhood Specific crime forecasting":
    st.subheader("Neighbourhood Specific crime forecasting")
    
    neighbourhoods = load_neighbours()
    if not neighbourhoods:
        st.warning("No neighbourhood models found. Please check the folder path and ensure it contains the correct model files.") 
        st.stop()
        
    selected_neighbours = st.sidebar.selectbox("Select a neighbourhood", options=neighbourhoods)
    try:
        model = load_neighbour_model(selected_neighbours)
        forecast = model.forecast(steps=months)
        plot_forecast(data=forecast,title=f"NEIGHBOURHOOD SPECIFIC CRIME FORECASTING FOR {months} MONTHS")
    except Exception as e:
        st.error(f"An error occurred while loading the model or forecasting: {e}")

##crime type specific forecasting
elif options == "Crime type specific forecasting":
    st.subheader("Crime type specific forecasting")
    
    crime_types = load_crime_types()
    if not crime_types:
        st.warning("No crime type models found. Please check the folder path and ensure it contains the correct model files.")
        st.stop()
        
    selected_crime_type = st.sidebar.selectbox("Select a crime type", options=crime_types)
    try:
        model = load_crime_model(selected_crime_type)
        forecast = model.forecast(steps=months)
        plot_forecast(data=forecast,title=f"CRIME TYPE SPECIFIC FORECASTING FOR {months} MONTHS")
    except Exception as e: 
        st.error(f"An error occurred while loading the model or forecasting: {e}")


###footer
st.markdown("")
st.write("""This app was developed by Pranay Raj Sambodhu using Streamlit and SARIMAX models for FBI crime forecasting. 
         The app allows users to forecast overall monthly crime, neighbourhood-specific crime, 
         and crime type-specific crime for a selected number of months.""")






























