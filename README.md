# Diabetes Predictor: An In-depth Analysis and Predictive Modeling

Authors:
Kush Momaya, Aditya Gambhir, Shaan Pakala, Harjyot Sidhu

*Full report available at: https://github.com/KushMomaya/CDC_Diabetes_Health_Indicators/blob/main/DSE_Final_Report.pdf

## Objective
The objective of the project was to develop a predictive model for diabetes using health indicators. We used various techniques to search for bias and unfairness to demonstrate ethical AI principles. 

## Dataset
The dataset we used comes from the CDC and contains 253,680 entries with 22 distinct health indicators. Chief among these is the Diabetes_binary variable which indicates the presence of diabetes in a patient and is 
our target variable. 

![image](https://github.com/user-attachments/assets/8202a742-92fa-4625-a48b-dcb559b43664)

We visualized the dataset using a correlation heatmap in order to glean some initial insights into the dataset and its features. We did this in order to isolate variables to guide our feature selection strategy with the goal of minimizing multicollinearity and mizimize interpretability.

## Data Preprocessing

To begin our preprocessing we scanned the data for any missing values and eliminated those entries from the dataset. 

We then identified several important variables such as BMI and GenHlth (General Health), and used the MinMaxScaler to normalize their values. This transforms the data to a range of values between 0 and 1 which equalized the scale of numerical variables. 

To address the categorical variables, we used encoding to make them machine readable so they could be processed by the model in its training process and predictions.

## Exploratory Data Analysis

Upon our initial analysis of the dataset, a significant issue we encountered was the lack of positive instances within the Diabetes_binary. To mitigate the issue caused by the high dimensionality, we used Principal Component Analysis (PCA) prior to initializing the model. 


