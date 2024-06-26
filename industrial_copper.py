import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu

# Functions
def predict_status(country,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,selling_price_log,
                   item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year):

    #change the datatypes "string" to "int"
    item_date_day= int(item_date_day)
    item_date_month= int(item_date_month)
    item_date_year= int(item_date_year)

    delivery_date_day= int(delivery_date_day)
    delivery_date_month= int(delivery_date_month)
    delivery_date_year= int(delivery_date_year)
    #modelfile of the classification
    with open("D:/HTML/classification_model.pkl","rb") as f:
        model_class=pickle.load(f)

    user_data= np.array([[country,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                       selling_price_log,item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year]])
    
    y_pred= model_class.predict(user_data)

    if y_pred == 1:
        return 1
    else:
        return 0
    
def predict_selling_price(country,status,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                          item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year):

    #change the datatypes "string" to "int"
    item_date_day= int(item_date_day)
    item_date_month= int(item_date_month)
    item_date_year= int(item_date_year)

    delivery_date_day= int(delivery_date_day)
    delivery_date_month= int(delivery_date_month)
    delivery_date_year= int(delivery_date_year)
    #modelfile of the classification
    with open("D:/HTML/Regression_Model.pkl","rb") as f:
        model_regg=pickle.load(f)

    user_data= np.array([[country,status,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                          item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year]])
    
    y_pred= model_regg.predict(user_data)

    ac_y_pred= np.exp(y_pred[0])

    return ac_y_pred

st.set_page_config(page_title= "Industrial Copper Modelling",
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This dashboard app is created by *Hannanstreamlit run  *!"""}
                  )

st.backgroundColor = '6739B7'

st.header(':violet[INDUTRIAL COPPER MODELLING]',anchor=False)
st.balloons()

with st.sidebar:
    st.title(":green[CAPSTONE PROJECT-6]")
    st.header("Introduction about Myself")
    st.caption("Name : Mohamed Hannan. S")
    st.caption("Course : Master in DataScience")
    st.caption("Batch : MDE88")

option = option_menu(
                menu_title = "Explore",
                options=["Home", "Selling Price","Status"],
                icons=["house-fill","database-fill","bar-chart-line"],
                default_index = 0,
                menu_icon="cast",
                orientation="horizontal",
                key="navigation_menu",
                styles={
                        "font_color": "#DC143C",   
                        "border": "2px solid #DC143C", 
                        "padding": "10px 25px"   
                    }
            )


if option == "Home":

    st.write('''
                # Industrial Copper Modelling
             ## Introduction:
             Copper Modelling is a Streamlit Application designed to predict the Selling price of the Product and the Status of it by using ML algorithms. The application allows users to enter the input for the feature columns from which the prediction is done. 

             ## Technologies Used:
             - Python: Programming language used for application development.
             - Streamlit: Web application framework for building interactive and customizable GUIs.
             - Pandas: Library for data manipulation and analysis.
             - NumPy: Library for numerical computing.
             - scikit-learn (sklearn): Machine learning library for classical algorithms in classification, regression, clustering, and more.
             - sklearn encoder: Refers to various encoders provided by scikit-learn, used for preprocessing categorical data in machine learning.
             - Matplotlib: Comprehensive library for creating static, animated, and interactive visualizations in Python.
             - Seaborn: Data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
             - Machine Learning (ML) classifier and regression algorithms: Algorithms from scikit-learn or other libraries used for building predictive models, including classifiers (e.g., Random Forest, SVM) and regression models (e.g., Linear Regression, Gradient Boosting).

             ## Problem Statement:
             The copper industry deals with less complex data related to sales and pricing. 
             However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. 
             Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. 
             A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, 
             and leveraging algorithms that are robust to skewed and noisy data. 

             ## Conclusion: 
                - Developing proficiency in Python programming language and its data analysis libraries such as Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

                - Gaining experience in data preprocessing techniques such as handling missing values, outlier detection to prepare data for machine learning modeling.

                - Understanding and visualizing the data using EDA techniques such as boxplots, histograms, and scatter plots.

                - Learning and applying advanced machine learning techniques such as regression and classification to predict target variables, respectively.

                - Building and optimizing machine learning models using appropriate evaluation metrics and techniques such as cross-validation and grid search.

                - Developing a web application using the Streamlit module to showcase the machine learning models and make predictions on new data.


             ''')


if option == "Status":

    st.header("PREDICT STATUS (Won / Lose)")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:
        country= st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        item_type= st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application= st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width= st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref= st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log= st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.322, Max:6.924",format="%0.15f")
        customer_log= st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910, Max:17.23015",format="%0.15f")
        thickness_log= st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.71479, Max:3.28154",format="%0.15f")
    
    with col2:
        selling_price_log= st.number_input(label="**Enter the Value for SELLING PRICE (Log Value)**/ Min:5.97503, Max:7.39036",format="%0.15f")
        item_date_day= st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month= st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year= st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day= st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month= st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year= st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE STATUS***]",use_container_width=True)

    if button:
        status= predict_status(country,item_type,application,width,product_ref,quantity_tons_log,
                               customer_log,thickness_log,selling_price_log,item_date_day,
                               item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                               delivery_date_year)
        
        if status == 1:
            st.write("## :green[**The Status is WON**]")
        else:
            st.write("## :red[**The Status is LOSE**]")

if option == "Selling Price":

    st.header("**PREDICT SELLING PRICE**")
    st.write(" ")

    col1,col2= st.columns(2)

    with col1:
        country= st.number_input(label="**Enter the Value for COUNTRY**/ Min:25.0, Max:113.0")
        status= st.number_input(label="**Enter the Value for STATUS**/ Min:0.0, Max:8.0")
        item_type= st.number_input(label="**Enter the Value for ITEM TYPE**/ Min:0.0, Max:6.0")
        application= st.number_input(label="**Enter the Value for APPLICATION**/ Min:2.0, Max:87.5")
        width= st.number_input(label="**Enter the Value for WIDTH**/ Min:700.0, Max:1980.0")
        product_ref= st.number_input(label="**Enter the Value for PRODUCT_REF**/ Min:611728, Max:1722207579")
        quantity_tons_log= st.number_input(label="**Enter the Value for QUANTITY_TONS (Log Value)**/ Min:-0.3223343801166147, Max:6.924734324081348",format="%0.15f")
        customer_log= st.number_input(label="**Enter the Value for CUSTOMER (Log Value)**/ Min:17.21910565821408, Max:17.230155364880137",format="%0.15f")
        
    
    with col2:
        thickness_log= st.number_input(label="**Enter the Value for THICKNESS (Log Value)**/ Min:-1.7147984280919266, Max:3.281543137578373",format="%0.15f")
        item_date_day= st.selectbox("**Select the Day for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        item_date_month= st.selectbox("**Select the Month for ITEM DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        item_date_year= st.selectbox("**Select the Year for ITEM DATE**",("2020","2021"))
        delivery_date_day= st.selectbox("**Select the Day for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"))
        delivery_date_month= st.selectbox("**Select the Month for DELIVERY DATE**",("1","2","3","4","5","6","7","8","9","10","11","12"))
        delivery_date_year= st.selectbox("**Select the Year for DELIVERY DATE**",("2020","2021","2022"))
        

    button= st.button(":violet[***PREDICT THE SELLING PRICE***]",use_container_width=True)

    if button:
        price= predict_selling_price(country,status,item_type,application,width,product_ref,quantity_tons_log,
                               customer_log,thickness_log,item_date_day,
                               item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                               delivery_date_year)
        
        
        st.write("## :green[**The Selling Price is :**]",price)
