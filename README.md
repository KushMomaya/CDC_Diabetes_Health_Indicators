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

![image](https://github.com/user-attachments/assets/7c8ab69a-fd79-46c0-8d2a-18b9c2afde1b)

In addition to this, we looked at the dataset from an ethical standpoint and identified several protected classes that could be culprits of bia and unfariness in the model. The most significant of these attributes were 'Sex', 'Age', 'Education', and 'Income'. We calculated the demographic parity difference and equalty of opportunity differences for these variables prior to our more in depth ethical analysis. 

## Model Preparation

We decided to use a stacking classifiers model to capture the insights of several diverse algorithms. This model uses multiple base classifiers and aggregates their results into a final classifier. Our base classifiers included: Logistic Regression, Random Forest, Multilayer Perceptron, XGBoost, and Gaussian Naive Bayes. Each of these captures and processes the data using diverse methods allowing us to encompass many aspects of our large dataset. 

![image](https://github.com/user-attachments/assets/e38e50fd-3454-494f-af05-528da818d428)

## Bias Mitigation - Fairness by Unawareness

We identified bias in our dataset and sought several different strategies to address it. Among these was Synthetic Minority Over-sampling Technique (SMOTE) which generated new instances of identified minority classes, balancing the model's variables. 

Another technique we used was threshold calibration, a post processing technique tht adjusted the decision thresholds of the model according to disparate impacts. Along with a complete model audit using Aequitas, we tackled issues of unfairness and bias in our model training process and results.

## Results and Conclusion

Our bias assessment found disparities caused by Age and Income, which we attempted to counter using different pre and post processing techniques. 

![image](https://github.com/user-attachments/assets/9ff4b1d2-4b95-4a40-ac8a-1ef798935681)
![image](https://github.com/user-attachments/assets/4ef24cb1-b323-48fe-8323-14c54eb2d3c9)

Throughout this analysis of sensitive health information, our process highlighted the symbiotic relationship between technical efficacy and ethical considerations. We prioritized fairness during our entire process and in doing so we were able to prove the efficacy of moral applications when it comes to AI and data related pursuits.
