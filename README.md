\documentclass[12pt]{report}

\usepackage{graphicx} % For including images
\usepackage{hyperref} % For hyperlinks
\usepackage{amsmath} % For mathematical expressions
\usepackage{geometry} % For page size and margins
\usepackage{float}
\geometry{a4paper, margin=1in} % Set margins

\title{
    {\includegraphics[width=4cm]{UCR Logo.png}}\\ % Add your university logo file here
    \vspace{0.5cm}
    {\fontsize{18}{22}\selectfont CS 108} \\
    \vspace{0.2cm}
    {\fontsize{18}{22}\selectfont Data Science Ethics} \\
    \vspace{0.2cm}
    {\large Fall 2023}
    
    \vspace{3cm}
    
    \hrule % Horizontal line before "Assignment Report"
    \vspace{0.5cm}
    {\fontsize{18}{22}\selectfont  \textbf{Project Report}} \\
    \vspace{0.2cm}
    {\Large Diabetes Predictor: An In-depth Analysis and Predictive Modeling} \\
    \vspace{0.5cm}
    \hrule % Horizontal line after "Assignment Report"
    
}
\author{
    Aditya Gambhir, Kush Momya, Shaan Palaka, Harjyot Sidhu \\
}
\date{\today}

\begin{document}

\maketitle


\section*{Introduction}

\subsection*{Objective}
The primary goal of this project is to develop a machine-learning model capable of accurately predicting whether a patient is at risk for diabetes based on a multitude of health-related attributes \cite{sonar2019diabetes}. In addition, we are committed to identifying and addressing any potential biases or unethical consequences that may arise from the application of our model, ensuring its fairness and integrity \cite{barocas2019fairness, hardt2016equality}.


\section*{Dataset}

\subsection*{Initial Description}
Our research project utilizes an extensive dataset derived from the Centers for Disease Control and Prevention (CDC), encompassing a wide array of health indicators across a substantial cohort. This dataset is comprised of 253,680 entries, each representing individual responses across 22 distinct health indicators, such as blood pressure, cholesterol levels, dietary habits, physical activity, and the presence of various health conditions\cite{cdc2023dataset}.

A notable aspect of this dataset is its meticulous organization and completeness, with no missing values across all entries, which facilitates a straightforward analysis process\cite{little2019statistical}. This integrity allows us to focus on the heart of our analysis—feature evaluation and model development without the need for extensive preprocessing to handle missing data.

The dataset encompasses both numerical and categorical variables, ranging from basic demographic information to detailed health-related metrics. For example, indicators like 'HighBP' (High Blood Pressure), 'HighChol' (High Cholesterol), and 'BMI' (Body Mass Index) are quantified alongside lifestyle factors such as smoking status and fruit and vegetable consumption. A critical variable within this dataset is 'Diabetes\_binary', a binary indicator denoting the presence or absence of diabetes, which serves as the target variable for our predictive modeling.

Statistical analysis of the dataset reveals a diverse range of values across the variables. For instance, the mean BMI of the cohort is 28.38, with a standard deviation of 6.61, indicating a wide variance in body weight. Similarly, the binary variables, such as 'Smoker' and 'Stroke', show a distribution reflective of the population's health characteristics. The 'Age' variable, treated categorically, spans from 1 to 13, representing various age groups, which could provide insightful correlations with the diabetes outcome.

This dataset's breadth and depth, characterized by its wide range of health indicators and the large sample size, offer a robust foundation for building a machine-learning model\cite{kavakiotis2017machine}. Our objective is to leverage this dataset to predict diabetes risk accurately while ensuring our model's fairness and ethical integrity by addressing and mitigating any potential biases inherent in the data.
\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{Correlation Heat Map.png}
\caption{Correlation Heatmap for Entire Dataset}
\end{figure}

\subsection*{Feature Analysis}
The analysis of features entailed a detailed examination of all 22 health indicators, aiming to comprehend their distribution patterns, identify any outliers, and investigate their correlation with the diabetes outcome. Through meticulous correlation analysis, we identified several features that exhibited significant associations with the outcome of diabetes, guided by recent findings in healthcare predictive analytics\cite{Badawy2023healthcare}. These findings guided our feature selection strategy, which was carefully designed to minimize multicollinearity and maximize the interpretability of our model.

\subsection*{Data Preprocessing}
Data preprocessing emerged as a crucial step in refining our dataset for the predictive modeling process. We embarked on this phase by performing a meticulous examination for missing values across the dataset and subsequently eliminated any such instances to guarantee a smooth and uninterrupted training phase. This proactive measure ensured the integrity and completeness of our data, setting a solid foundation for the subsequent stages of model preparation.

Following the initial cleanup, we directed our focus toward the normalization of a selected subset of features, specifically targeting variables that significantly influence the model's learning capacity and predictive accuracy\cite{sklearn_preprocess}. Utilizing the MinMaxScaler, we normalized the values of key variables including 'BMI', 'GenHlth' (General Health), 'MentHlth' (Mental Health), 'PhysHlth' (Physical Health), 'Age', 'Education', and 'Income'. This scaling process transformed the selected features to a uniform range between 0 and 1, thereby facilitating a more effective learning process by equalizing the scale of numerical inputs.

In the final step of our preprocessing journey, we addressed the categorical variables within our dataset. Through the application of appropriate encoding techniques, we transformed these categorical variables into a machine-readable format, enabling their utilization in predictive modeling\cite{arkon_data_preprocess}. This transformation is crucial for incorporating the rich, categorical information contained within our dataset into the predictive modeling process, further enhancing the model's ability to discern and learn from complex patterns and relationships in the data.

Collectively, these preprocessing steps—ranging from the elimination of missing values and normalization of numerical features to the encoding of categorical variables—have meticulously prepared our dataset for the intricate demands of machine learning model development. This comprehensive approach to data preparation not only facilitates a more robust and effective modeling process but also underscores our commitment to leveraging data in a manner that maximizes predictive accuracy and model reliability\cite{ml_mastery_data_prep}.

\section*{Exploratory Data Analysis}

\subsection*{General EDA}
In the exploratory phase of our analysis, we embarked on a detailed exploration of the dataset to identify underlying patterns or significant predictors for the `diabetes\_binary` outcome. Initial visualizations were focused on variables anticipated to be crucial in predicting diabetes risk. 

A notable observation was the presence of disparities, particularly related to the `Sex` attribute, which we marked for further evaluation regarding its impact on model bias and fairness. Additionally, we encountered a pronounced scarcity of positive instances within the dataset, a crucial factor that warranted subsequent adjustment strategies. To mitigate the challenge posed by the dataset's high dimensionality and facilitate a more efficient model fitting, Principal Component Analysis (PCA) was implemented as a preliminary step before model initialization\cite{Nedyalkova2020}.

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Sex General EDA.png}
\caption{Sample of EDA}
\end{figure}

\subsection*{Protected Attributes}
Our investigation into protected attributes highlighted significant disparities that could potentially influence the model's fairness and bias. The analysis was conducted across multiple attributes, namely `Sex`, `Age`, `Education`, and `Income`, with each showing varying levels of impact on the model's predictions.

\textbf{`Sex` Attribute Analysis:}
The Disparate Impact Analysis yielded a ratio of 0.8554 (Ideal: 1), indicating a notable disparity. Specifically, the positive outcomes ratios for Groups 0 and 1 were 0.9307 and 1.0881, respectively, suggesting an imbalance that merits attention for fairness corrections. Demographic Parity Differences and Equality of Opportunity Differences were also computed, showing deviations from the ideal values, thereby confirming the presence of bias in this attribute\cite{Brookings2020fairness}.
\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Sex EDA Protected.png}
\caption{Count plot for Sex Attribute}
\end{figure}

\textbf{`Age` Attribute Analysis:}
The analysis of the `Age` attribute revealed a Disparate Impact Analysis ratio of 0.0626, significantly deviating from the ideal. This analysis detailed the progression of positive outcomes ratios across age groups, illustrating a clear trend that could affect the model's fairness. The Demographic Parity Differences and Equality of Opportunity Differences across all age groups consistently highlighted disparities\cite{Karthik2019mitigating}, emphasizing the need for careful consideration of age as a protected attribute in our model.
\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Age EDA Protected.png}
\caption{Count plot for Age Attribute}
\end{figure}

\textbf{`Education` Attribute Analysis:}
For the `Education` attribute, a Disparate Impact Analysis ratio of 0.3312 was observed, indicating potential bias\cite{Hao2018}. The positive outcomes ratios across educational groups varied widely, with higher education levels generally correlating with better outcomes. This trend underscores the influence of education on model predictions and highlights the importance of adjusting for such biases.
\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Education EDA Protected.png}
\caption{Count plot for Education Attribute}
\end{figure}

\textbf{`Income` Attribute Analysis:}
The `Income` attribute's Disparate Impact Analysis ratio stood at 0.3039, suggesting significant disparities based on income levels. The analysis showed a gradient in positive outcomes ratios, with higher income groups generally experiencing better predictions. These findings necessitate adjustments in the model to mitigate income-based disparities and ensure equitable predictions across all income groups.
\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Income EDA Protected.png}
\caption{Count plot for Income Attribute}
\end{figure}

\section*{Stacking Ensemble}

\subsection*{Model Preparation}
\subsubsection*{Principal Component Analysis(PCA)}
The implementation of PCA in our project was pivotal for dimensionality reduction and feature extraction. To retain significant variance while transforming the feature space, PCA reduced the dataset to two principal components\cite{Lever2017}. 

The cumulative explained variance by these components was 25\%\cite{Lever2017}, as evidenced by the following plot:

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{PCA Variance.png}
\caption{Explained Variance by PCA Components}
\end{figure}

Moreover, we visualized the dataset in the reduced two-dimensional space, enhancing the aesthetics to aid interpretability. The following scatter plot illustrates the distribution of the dataset across the first two principal components, color-coded by the diabetes label:

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{PCA Scatter Plot.png}
\caption{PCA of Dataset with Enhanced Aesthetics}
\end{figure}

To interpret the principal components, we examined the PCA loadings\cite{Lever2017}, which reflect the correlation between the original features and the components. The following bar chart showcases the loadings for the first principal component, highlighting the influence of each feature on this component:

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Loadings PCA.png}
\caption{Loadings for Component 1}
\end{figure}

\subsection*{Stacking Classifiers}
In this segment, we elucidate the construction and training of a stacking ensemble model, designed to harness the collective prowess of diverse machine learning algorithms for our binary classification task. Stacking, or stacked generalization\cite{Wolpert1992}, is an ensemble learning paradigm that amalgamates the predictions from multiple base classifiers via a final estimator, thereby amplifying the predictive acumen of the model.

\textbf{Base Classifiers:} We commence with the instantiation of an array of classifiers, each equipped with a meticulously selected suite of hyperparameters. These classifiers, functioning as the foundational learners in our stacking ensemble, include:
\begin{itemize}
    \item \textit{Logistic Regression (LR)}\cite{Defazio2014}: A linear classification model employing L1 regularization and the SAGA solver for optimization purposes.
    \item \textit{Random Forest (RF)}\cite{Breiman2001}: An ensemble technique founded on decision trees, which utilizes the Gini impurity criterion for node splitting and a square root function for feature selection at each bifurcation.
    \item \textit{Multi-layer Perceptron (NN)}\cite{Hinton2012}: A neural network architecture comprising two hidden layers, deploying the ReLU activation function and an adaptive learning rate scheme.
    \item \textit{XGBoost (XGB)}\cite{Chen2016}: A gradient boosting framework underpinned by decision trees, with hyperparameters, fine-tuned to mitigate overfitting and bolster model efficacy.
    \item \textit{Gaussian Naive Bayes (NB)}\cite{Rish2001}: A probabilistic classifier predicated on Bayes' theorem, with an underlying assumption of feature independence.
\end{itemize}
The selection of these classifiers is predicated on their ability to capture a diverse array of patterns from the data, thereby encompassing linear, ensemble, neural network, and probabilistic approaches.

\textbf{Final Estimator:} The ensuing phase involves the base classifiers' predictions (probability estimations) being utilized as input for the final estimator—another XGBoost model\cite{Breiman1996}. This final estimator is tasked with the optimization of the base classifiers' combined predictive output.

\textbf{Stacking Ensemble Classifier:} We employ the \texttt{StackingClassifier} from the \texttt{scikit-learn} ensemble module to synthesize the individual classifiers with the final estimator into a unified model\cite{scikit-learn}. The \texttt{stack\_method='predict\_proba'} parameter is set to leverage the base classifiers' probabilistic predictions for the training of the final estimator, which is essential for binary classification tasks.

\textbf{Training the Model:} The ensemble classifier is trained on the complete training dataset, with each base classifier learning from the data and the final estimator being trained on their predictions. This training regimen is computationally demanding, entailing the sequential training of multiple models. Consequently, we monitor and report the duration of the training to evaluate performance.

The objective of this stacking ensemble model is to amalgamate the individual strengths of each classifier while counterbalancing their inherent limitations, culminating in a robust and precise model. The convergence of these disparate models is anticipated to deliver superior predictive performance compared to the potential of a solitary classifier.
\begin{figure}[H]
\centering
\includegraphics[width=1\textwidth]{Stacking Classifier.png}
\caption{Trained Stacking Classifer}
\end{figure}

\section*{Bias Detection and Mitigation Strategies}

\subsection*{Fairness by Unawareness}
In pursuit of fairness through unawareness, we adapted our dataset to curtail potential biases, thus fostering a more equitable representation of classes\cite{Dwork2012}. We explored dual strategies: adjusting the training data by excising samples, and amplifying the minority class through oversampling\cite{Chawla2002}. The chosen tactics, grounded in the configuration of boolean flags—adjust\_train\_data and over-sample—yielded two distinct datasets. 

Our process initiated with the original dataset, from which we derived a modified copy, designed to undergo bias mitigation. Should adjust\_train\_data be enacted without oversampling, we deliberately reduced the majority class instances, thereby attenuating class imbalance. Post-adjustment, we scrutinized the dataset, affirming its alignment with fairness principles by excluding sensitive attributes such as Sex, Age, Education, and Income, thus safeguarding against their undue influence in model training. 

\subsection*{Balancing Target Class Subgroups}
The imbalance within target classes posed an additional bias risk. To counteract this, we employed the Synthetic Minority Over-sampling Technique (SMOTE), which synthetically generated new instances of the minority class, thereby presenting a balanced perspective of the classes to the model\cite{Chawla2002}. In scenarios where both adjust\_train\_data and over-sample flags were activated, we further refined the training data, ensuring the minority class outnumbered the majority, compelling the model to accord greater consideration to the former. 

This comprehensive approach to data preparation was visually validated; histograms depicting the class distribution before and after adjustments elucidated the alterations made to achieve class balance. 

\subsection*{Threshold Calibration}
Post-detection of biases within the dataset, we resorted to threshold calibration, a post-processing technique, to mitigate biases across various classes\cite{hardt2016equality}. By adjusting the decision thresholds according to the disparate impacts identified, we aspired to realize a more nuanced balance of model predictions across demographic groups.

\subsection*{Model Performance Audit}
A rigorous audit of the model's performance, encompassing sensitive attributes, provided insights into the bias landscape\cite{Saleiro2018}. This was accomplished by appending sensitive attributes post-model training, enabling subgroup metrics analysis. The evaluation underscored the essence of threshold calibration, with performance metrics indicating improved fairness in the model's predictive capabilities.

\newpage
\section*{Results and Conclusion}

\textbf{Results Summary}\\
Our multifaceted exploration into diabetes prediction culminated in the construction of a stacking ensemble model\cite{Ting1999}. This model judiciously integrates the predictive prowess of Logistic Regression, Random Forest, Multi-layer Perceptron, XGBoost, and Gaussian Naive Bayes classifiers. The ensemble's performance, adjudicated by a secondary XGBoost classifier, yielded commendable results, as delineated by the metrics tables below\cite{Zhou2020prediction}.

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\hline
Metric & Basic Model & Masked Model \\ \hline
Accuracy & 0.8666 & 0.8662 \\
Precision & 0.5686 & 0.5717 \\
Sensitivity (Recall) & 0.1450 & 0.1296 \\
Specificity & 0.9823 & 0.9844 \\
F1-Score & 0.2311 & 0.2113 \\
Time Taken (s) & 2465.1 & 722.8 \\ \hline
\end{tabular}
\caption{General Metrics for Basic and Unawareness Models}
\label{table:general_metrics}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\hline
Metric & Basic Classifier & Masked Classifier \\ \hline
True Positive Rate (TPR) & 0.1450 & 0.1296 \\
False Positive Rate (FPR) & 0.0177 & 0.0156 \\
False Negative Rate (FNR) & 0.8550 & 0.8704 \\
Positive Predictive Value (PPV) & 0.5686 & 0.5717 \\
Negative Predictive Value (NPV) & 0.8775 & 0.8758 \\
Brier Score & 0.0964 & 0.0981 \\ \hline
\end{tabular}
\caption{Detailed Metrics for Basic and Unawareness Classifiers}
\label{table:detailed_metrics}
\end{table}

Bias assessment unearthed disparities in subgroup metrics, particularly across Age and Income brackets. Measures to counteract this, including the exclusion of sensitive attributes in the training phase, led to significant modifications in model predictions. These were aimed at fostering a balanced representation of diabetes risk, as depicted by the Aquetas plots and calibration curves for both full and masked datasets. \cite{Kiyasseh2021BiasMitigation, Char2021HealthEquity}


\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{Aquetas Age Unawareness.png}
\caption{Subgroup Metrics by Age for the Basic Model}
\label{fig:aquetas_age_unawareness}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{Aquetas Age.png}
\caption{Subgroup Metrics by Age for the Unawareness Model}
\label{fig:aquetas_age}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{Aquetas Income.png}
\caption{Subgroup Metrics by Income for the Basic Model}
\label{fig:aquetas_income}
\end{figure}

Calibration plots for the full dataset and the masked dataset illustrate the model's performance against the ideal of perfect calibration. Both plots demonstrate the models' fidelity in predicting diabetes presence, with the masked dataset model showcasing a slight deviation from perfect calibration. \cite{Brownlee2020Calibration}

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Caliberation Curve Full Dataset.png}
\caption{Calibration Plot for the Full Dataset Model}
\label{fig:calibration_full}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.6\textwidth]{Caliberation Curve Masked Dataset.png}
\caption{Calibration Plot for the Masked Dataset Model}
\label{fig:calibration_masked}
\end{figure}

Confusion matrices further delineate the accuracy of predictions for both the full and masked dataset models, emphasizing the modifications enacted post-bias mitigation strategies. \cite{Brownlee2020Calibration}

\begin{figure}[H]
\centering
\includegraphics[width=0.5\textwidth]{Confusion Matrix Full.png}
\caption{Confusion Matrix for the Full Dataset Model}
\label{fig:confusion_full}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.5\textwidth]{Confusion Matrix Masked.png}
\caption{Confusion Matrix for the Masked Dataset Model}
\label{fig:confusion_masked}
\end{figure}

\textbf{Analysis of Findings}\\
Our investigation into diabetes risk prediction has emphasized the crucial balance between achieving model accuracy and ensuring fairness. Adjustments implemented to rectify imbalances and bias incurred a measurable impact on precision—a testament to the ethical conundrum faced in predictive modeling. This necessary trade-off underscores the criticality of equitability in algorithmic decision-making within healthcare, where biases can have significant ramifications. The project's discoveries are a clarion call to meticulously scrutinize and prepare data, advocating for ethical integrity in health-related AI applications.

\textbf{Ethical Considerations}\\
This endeavor is a reaffirmation of our commitment to embedding ethical tenets within the realm of data science and machine learning. The proactive identification and rectification of bias in our model iterate the broader dialogue on AI's ethical use in healthcare. By championing transparency, equity, and accountability, we endeavor to establish a precedent for the ethical deployment of AI, ensuring its implications are constructive and universally beneficial.

\textbf{Future Work}\\
The foundation established by our project invites further inquiry into fairness in predictive modeling. Subsequent efforts can build upon our methodologies, harnessing expansive and varied datasets, and broadening the application scope to other medical conditions. This forward-looking approach aims to guarantee that AI's advancements are equitably shared, thereby fostering an inclusive future where AI serves the collective good.

\textbf{Conclusion}\\
In summation, our venture has highlighted the symbiotic relationship between technical efficacy and moral responsibility in developing healthcare AI. By prioritizing fairness and diligently countering bias, we establish that ethical AI is an attainable reality, achieved through meticulous analysis and deliberate practice. Our contributions advocate for a just and equitable application of technology in healthcare—a paradigm where models are not merely tools but bearers of integrity, aiding every individual impartially.

\bibliographystyle{ieeetr}
\bibliography{references.bib}

\end{document}

