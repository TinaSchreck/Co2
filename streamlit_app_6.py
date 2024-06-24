# STREAMLIT TO DEMONSTRATE WORK ON CO2 EMISSIONS BY CARS IN FRANCE 2015

# necessary imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# read the dataframe
df=pd.read_csv("cleaned_data_France_2015.csv")

# to add a title and to create 8 pages
st.title("CO2 emission of cars in France 2015")
st.sidebar.title("Table of contents")
pages=["Cover", "Introduction", "Exploration", "DataVisualization", "Modeling Regression", "Preparing for Classification", "Modeling Classification", "Conclusion"]
page=st.sidebar.radio("Select page", pages)

# work on cover page ##############################################################################################
if page == pages[0] : 
  st.text("")
  st.text("")
  st.text("")
  image_path = ("car smog.jpg")
  # Display the image centered
  st.image(image_path, use_column_width=True, caption='Cars in traffic')

# work on first page ##############################################################################################
if page == pages[1] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Introduction")
  st.write("""
    The importance of CO2 emissions from cars:
    - Cars and vans account for almost half of global transport carbon dioxide emissions, according to an analysis by Statista. 
    - France started collecting data on cars and their CO2 emissions in 2001.
    - The latest dataset we had is from 2015. 
    - The objective of this project is the analysis of the influence of the different variables (brand, horsepower, weight etc) on CO2-emissions. 
    """)
  
  image_path = ("cars_48%.jpeg")
  # Display the image centered
  st.image(image_path, use_column_width=True, caption='Pie Chart of CO2 emissions')

# work on second page #############################################################################################
if page == pages[2] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Presentation of data")
  st.write('')
  st.dataframe(df.head(10))
  st.write('Shape of the dataframe:',df.shape[0],'rows and',df.shape[1],'columns')
  st.write('')                
  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())
  st.write('')
  st.subheader('Statistics of dataframe')
  st.dataframe(df.describe())


# work on third page ##############################################################################################
if page == pages[3] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### DataVisualization")
  st.text("")  # Adds a line break
  st.text("")
  st.text("")

# distribution of car brands
  st.image("distribution_car_brands.png")
  st.text("")
  st.text("")

# top 5 highest emissions according to brands 
  st.image("top_5_highest.png")
  st.text("")  # Adds a line break
  st.text("")

# top 5 lowest emissions according to brands 
  st.image("top_5_lowest.png")
  st.text("")  # Adds a line break
  st.text("")

  # display heatmap of correlation for the numeric columns
  st.image("heatmap.png")
  st.text("")
  st.text("")

  # display statistics
  st.markdown("### STATISTICS")
  st.text("")

  # Display QQ plots for normal distribution
  st.markdown("### QQ Plots for Normal Distribution:")
  st.text("")
  st.text("")
  st.image("qqplot_1.png")
  st.text("")
  st.text("")
  st.image("qqplot_2.png")
  st.text("")
  st.text("")
  st.image("qqplot_3.png")
  st.text("")
  st.text("")
  st.image("qqplot_4.png")
  st.text("")
  st.text("")
  st.image("qqplot_5.png")
  st.text("")
  st.text("")
  st.image("qqplot_6.png")
  st.text("")
  st.text("")
  st.image("boxplot energy classes.png")


# work on fourth page ###############################################################################################
if page == pages[4] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Modeling Regression")

  # preparation of dataset
  X = df.drop(['CO2 in g/km', 'Brand'], axis=1)
  y = df['CO2 in g/km']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # define models
  files = ['DecisionTreeRegressor().joblib', 'KNeighborsRegressor().joblib', 'LinearRegression().joblib', 'LogisticRegression().joblib']
  names = ['Decision Tree', 'KNeighbors', 'Linear Regression', 'Logistic Regression']

  reg_models = []
  
  # load models and add to list
  for file, name in zip(files, names):
    model = load(file)
    reg_models.append((name,model))

  # choice of regression model
  selected_regression_model = st.selectbox('Choice of regression model:', names)
  st.write(f'The choosen regression model is: {selected_regression_model}')

  # perform analysis after model selected
  if selected_regression_model:
    #finding choosen model
    selected_model = None
    for name, model in reg_models:
      if name == selected_regression_model:
        selected_regression_model = model
        break

  # perform analysis after chosen model found
  if selected_regression_model:
    y_pred = selected_regression_model.predict(X_test)


# work on fifth page ##############################################################################################
if page == pages[5] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Preparing for Classification")
  st.subheader("")
  st.subheader("Transforming CO2 emission into a categorical variable")
  st.text("")
  st.subheader("(1) Calculation of reference value")
  st.write("R [g/km] = 36,59079 + 0,08987 × m [kg]")
  st.text("")
  st.subheader("(2) Calculation of difference in CO2 emission in percent")
  st.write("CO2Diff. [%] = (CO2Pkw [in kg from Dataframe] – R) × 100 / R ")
  st.text("")
  st.subheader("(3) Group into following CO2 emission classes")
  data = {'CO2 emission class': ['A+++', 'A++', 'A+','A','B','C','D','E','F','G'],
    'CO2"Diff[%]"': ['x ≤ -55', '-55 < x ≤ -46', '-46 < x ≤ -37', '-37 < x ≤ -28', '-28 < x ≤ -19', '-19 < x ≤ -10', '-10 < x ≤ -1', '-1 < x ≤ 8', '8 < x ≤ 17', '> 17']}
  # Create a DataFrame
  df = pd.DataFrame(data)
  # Display the table
  st.table(df)
  st.text("")
  st.subheader("(4) Show results")
  checkbox_class = st.checkbox('Distribution of transformed CO2 emission classes')
  if checkbox_class:
    st.image('Distribution_emission_classes.jpg')

    
# work on sixth page##############################################################################################
if page == pages[6] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Modeling Classification")


# work on seventh page#################################################################################################
if page == pages[7] : 
  st.text("")
  st.text("")
  st.text("")
  st.write("### Conclusion")
  st.write("""
    A look into the future:
    - The excellent modeling results might indicate overfitting, and we would have liked to do more tests as well as comparisons with other datasets.
    - A proper prediction model for future years would be rather complicated, trying to take into account sinking CO2 emissions due to an increase in electric cars, then projecting that tendency. 
    - We would like to do that if time permits, because the topic in itself is very important: after all, this is about saving our planet.  
    - An accurate model might allow a hopeful outlook or increase the urgency to further cut down on CO2 emissions. 
    """)
