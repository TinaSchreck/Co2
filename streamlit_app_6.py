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

  # separation of dataset into training and test set and deleting the variable Brand
  X = df.drop(['CO2 in g/km', 'Brand'], axis=1)
  y = df['CO2 in g/km']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # transformation of test set
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # define names and of files and models
  files = ['DecisionTreeRegressor().joblib', 'KNeighborsRegressor().joblib', 'LinearRegression().joblib','LogisticRegression().joblib']
  names = ['Decision Tree', 'KNeighbors', 'Linear Regression','Logistic Regression']

  reg_models = []

  # load models and add to list 
  for file_name, name in zip(files, names):
    model = load(file_name)
    reg_models.append((name, model))

  # choice of classification model
  selected_model_name = st.selectbox('Choice of regression model', names, key='choice_reg_model')
  ('The chosen regression model is:', selected_model_name)

  # perform analysis after an option has been selected
  if selected_model_name:
    # finding the chosen model
    selected_model = None
    for name, model in reg_models:
        if name == selected_model_name:
            selected_model = model
            break
    
    # perform analysis after chosen model has been found
    if selected_model:
        # analysis for model
        #(f"Model '{selected_model_name}' geladen:", selected_model)
        
        # code for analysis
        y_pred = selected_model.predict(X_test)
        #st.write('r_squared of', model,'on test set:',round(r2_score(y_test, y_pred),4))
        #st.write('Mean Squared Error (MSE) on', model,':', round(mean_squared_error(y_test, y_pred),2))

  def scores(reg_model,choice):
    if choice == 'R2':
      return reg_model.r2_score(y_test,y_pred )
    elif choice == 'MSE':
      return mean_squared_error(y_test,y_pred) 

  choice = model
  display = st.radio('What do you want to show ?', ('R2', 'MSE'))
  if display == 'R2':
    st.write(round(r2_score(y_test,y_pred),3))
  elif display == 'MSE':
    st.write(round(mean_squared_error(y_test,y_pred),3))
    
  st.subheader("Short overview of real and predicted values")
  y_pred = pd.DataFrame(y_pred)
  y_pred.rename(columns={0: 'Predicted CO2 emissions in g/km'}, inplace=True)
  #st.dataframe(y_pred.head(11))

  #st.text("short overview of real values")
  y_test = pd.DataFrame(y_test)
  y_test.rename(columns={0: 'Real CO2 emissions in g/km'}, inplace=True)
  #st.dataframe(y_test.head(11))

  st.write(
    f"""
    <div style="display:flex">
        <div style="flex:50%;padding-right:10px;">
            {y_test.head(11).to_html()}
        </div>
        <div style="flex:50%">
            {y_pred.head(11).to_html()}
        </div>
    </div>
    """,
    unsafe_allow_html=True
  )
  
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

  ######FOR SAVING MODELS
  #from joblib import dump
  # saving of the model(after training)
  # dump(model, 'filename.joblib')
  # loading of the model (via streamlit app)
  #model = load('filename.joblib')

  #def load_files_from_folder(folder_path, file_extension):
  #    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
  #   return files

  #def load_files_from_folder('\documents', joblib):
  #   files = [f for f in os.listdir('\documents') if f.endswith(joblib)]
  #  return files
   
  df = pd.read_csv('df_class.csv')


  # separation of dataset into training and test set and dropping not needed variables for classification
  X = df.drop(['Unnamed: 0','Urban consumption in l/100km','Rural consumption in l/100km','energy'], axis=1)
  y = df['energy']

  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # transformation of test set
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # define names of files and models
  files = ['DecisionTreeClassifier().joblib', 'RandomForestClassifier().h5', 'SVC().joblib']
  names = ['Decision Tree', 'Random Forest', 'SVC']

  class_models = []

  # load models and add to list
  for file_name, name in zip(files, names):
    model = load(file_name)
    class_models.append((name, model))

  # choice of classification model
  selected_model_name = st.selectbox('Choice of classification model', names, key='choice_class_model')
  st.write('The chosen classification model is:', selected_model_name)

  # perform analysis after a model has been chosen
  if selected_model_name:
    # finding the chosen model 
    selected_model = None
    for name, model in class_models:
        if name == selected_model_name:
            selected_model = model
            break
    
    # perform analysis if model has been found
    if selected_model:
        # perform analysis for the model
        #st.write(f"Model '{selected_model_name}' geladen:", selected_model)
        
        # code for analysis
  
      y_pred = selected_model.predict(X_test)
      accuracy = round(selected_model.score(X_test, y_test), 2)
      st.write('Accuracy of', selected_model_name, 'on test set:', accuracy)

  ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
  st.write('The confusion matrix of', selected_model_name, 'is:')
  st.write(ct)
  
  report = classification_report(y_test, y_pred, output_dict=True)
  print(report)

  # Convert classification report to DataFrame for easier manipulation
  report_df = pd.DataFrame(report).transpose()

  # Display the classification report as a table
  st.write('The classification report  of', selected_model_name, 'is:')
  st.dataframe(report_df)

  #st.write('The classification report of', selected_model_name, 'is:')
  #st.write(classification_report(y_test, y_pred))

  st.subheader("Short overview of real and predicted values")
  y_pred = pd.DataFrame(y_pred)
  y_pred.rename(columns={0: 'Predicted CO2 emission classes'}, inplace=True)
  #st.dataframe(y_pred.head(11))
  
  #st.text("short overview of real values")
  y_test = pd.DataFrame(y_test)
  y_test.rename(columns={0: 'Real CO2 emission classes'}, inplace=True)
  #st.dataframe(y_test.head(11))

  st.write(
    f"""
    <div style="display:flex">
        <div style="flex:50%;padding-right:10px;">
            {y_test.head(11).to_html()}
        </div>
        <div style="flex:50%">
            {y_pred.head(11).to_html()}
        </div>
    </div>
    """,
    unsafe_allow_html=True
  )

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
