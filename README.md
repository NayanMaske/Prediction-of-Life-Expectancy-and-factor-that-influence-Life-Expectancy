# Prediction-of-Life-Expectancy-and-factor-that-influence-Life-Expectancy
1.Data description and research question:
1.1Introduction:
The World Health Organization's (WHO) Global Health Observatory (GHO) data repository keeps track of all countries' health status as well as many other related factors. many studies have been conducted in the past on factors influencing life expectancy, including demographic variables, income composition, and mortality rates. It was discovered that the impact of immunization and the human development index was not previously considered. Since the observations in this dataset are based on different countries, it will be easier for a country to determine the predicting factor which is contributing the to lower value of life expectancy. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors, and other health-related factors as well. This will help in suggesting a country which area should be given importance to efficiently improve the life expectancy of its population.

1.2 Dataset Description:
In the last 30 years, there has been significant progress in the health sector, resulting in lower human mortality rates, particularly in developing countries. As a result, in this project, we considered data from 2000 to 2015 for 193 countries for further analysis. The data set related to life expectancy, and health factors for 193 countries have been collected from the Kaggle website. Dataset consists of 22 Columns and 2938 rows which meant 21 predicting variables. All predicting variables were then divided into several broad categories:Immunization related factors, Mortality factors, Economical factors, and social factors .in the dataset Life expectancy is the target variable, and the rest of all are the independent variable. The dataset is downloaded from Kaggle https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who


1.3 Research Question:
1.	Do the various predicting factors that were initially chosen to have any effect on life expectancy? What are the factors that influence life expectancy?

2. Data preparation and cleaning:
For data preparation and cleaning, the following steps are taken:
1.	Import necessary R packages and libraries
2.	Load Data frame in R markdown
3.	Inspect data frame
4.	Detecting missing values
5.	Calculate the Na Value in each column.
6.	Calculate the percentage of Na Value in each column.
7.	Imputing missing values
8.	Outlier detection
9.	Removing outliers
1.	Importing necessary R packages and libraries: - importing all the necessary packages and libraries in R Markdown in this step.
2.	Loading Data frame in R markdown – data frame loaded and stored in the “Life” object
3.	Inspect-Inspect the data frame by using the str(Life) command.
4.	Detecting missing values - by using the summary () function and further research into the NA value of each column with their percentage
5.	Calculation - Calculating Na values in each column
6.	Calculation - Calculating the percentage of NA value of each variable.
7.	Imputing NA values - At this stage, I found some histograms are left-skewed, and some are right-skewed. So, I used Median to replace the Na value. As the median does not affect the data.
8.	Outlier Detection – outliers are detected by using Boxplot.
9.	Outliers Removal- At this step outliers are removed by using the Winsorizing technique to improve prediction Outliers were removed by deleting them from the Life expectancy column.
 
Fig 2: Winsorizing Technique
                                                             

 
 

3. Exploratory data analysis
A status variable is converted into a categorical variable by using the dummy variable ‘developed’.In this step, correlations are checked between variables by using corrplot () and correlation function(), cor(NewLife[4:22]).After all the data cleaning process 2919 rows and 23 variables remain.
Observations:
•	There are 82% of countries status are Developing and 18% of countries are developed.
•	Through the graph, we can observe life expectancy is low in developing countries as compared to developed countries.
•	Highly correlated variables with Life expectancy are Alcohol, Percentage expenditure, BMI, Polio, Diphtheria, GDP, Income composition of resources, and Schooling.
•	Correlated variables with Life expectancy are Hepatitis B. Measles, total Expenditure,
•	No correlation with  Under-five deaths, HIV/AIDS, Thinness 1.9 years, Thinness 5.9 years,Adult mortality, Infant deaths, population.
•	Diphtheria is highly correlated with polio.
•	Thinness’s 1.9 years is highly correlated with Thinness’s 5.9 years
•	Infant deaths are highly correlated with under-five deaths.
•	Percentage expenditure is highly correlated with GDP
•	Polio is correlated with diphtheria
•	Income composition resources is highly correlated with schooling
Over time, life expectancy has increased.
•	In developed countries, the adult mortality rate, the prevalence of thinness in children aged 10 to 19 years, infant mortality, and HIV deaths are all high. Because AIDS is less prevalent in developed countries than in developing countries, developed countries have a higher life expectancy.
•	Other variables, such as resource income composition, number of years of schooling, percentage expenditure, and total health expenditure, are higher in developed countries than in developing countries, raising life expectancy in developed countries.
•	Furthermore, increased immunization coverage against Hepatitis B, Polio, and Diphtheria in developed countries results in higher life expectancy compared to developing countries.
•	Based on the preceding steps, we can conclude that immunization factors, economic factors, social factors, and other health-related factors have some effect on life expectancy, whereas mortality factors have no effect on life expectancy.
All of the preceding steps are completed in Rmarkdown with R programming. Pyspark was used to implement and analyze machine learning algorithms.


4. Machine learning prediction:
After EDA step Unsupervised learning method is used. (PCA)
Why PCA?
PCA is used for dimensionality reduction. A dataset with a high dimensionality has many features. Model overfitting is the primary issue associated with high dimensionality in the machine learning field. There are 22 variables in Dataset, plus one dummy variable for a status column. As a result, PCA is a good choice to reduce dimensionality and work with a new dataset.
Following are the observations of PCA-
we obtained 19 components, most of the dataset information was captured by the first seven or eight PCs. In addition, the variance plot displays seven components that can be used in further modelling.

 

After using PCA, I moved on to PySpark to model machine learning. A clean RStudio data file export and the same file import in PySpark for further Machine learning Modeling decision tree machine learning algorithm is used to predict a country's life expectancy and the factors that influence a country's life expectancy. 
The following steps are performed for Random Forest Regression Analysis
•	In PySpark imported dataset is stored in the df object.
•	First, processed data is split into a training set and test set in the ratio of 70:30
•	The training set is stored in the train object whereas the Test set is stored in a test object
 
•	Import Random Forest regressor.
•	Train the model by using the fit () method
•	Evaluate () method is used to predict the life expectancy by using a test dataset.
•	Find out the Accuracy and print it beside the feature column.  

•	Import the decision tree regressor package and find the Decision Tree summary RMSE, MSE, MAE, R2
•	The final output of the model is
 

5. High-Performance Computational implementation
PySpark is being used for implementation in this case. Python is a general-purpose, high-level programming language, whereas Apache Spark is an open-source cluster-computing framework focused on speed, ease of use, and streaming analytics. It offers a diverse set of libraries and is primarily used for Machine Learning and Real-Time Streaming Analytics. This is the first part of PySpark (Python, n.d.)
SparkSession is an entry point to Spark to work with RDD, DataFrame, and Dataset.
 
If Spark Session already exists, it returns otherwise create a new Spark Session.
The following steps are performed:
•	New Life file which is exported is imported here through HDFS with the name of DDA
•	DDA.csv file read and stored in df object.
•	Created numeric features object in which all the numeric variables are stored.
•	Categorical cols are created to store categorical columns (Country and Status)
•	A correlation heatmap is generated to find correlated factors.
•	Next step OneHotEncoder is used to convert the categorical column into numeric.
A categorical feature, represented as a label index, is converted to a binary vector with at most a single one-value indicating the presence of a specific feature value from the set of all feature values by one-hot encoding. This encoding enables categorical features to be used by algorithms that expect continuous features, such as Logistic Regression. When dealing with string-type input data, it is common to practice first encoding categorical features with String Indexer. For each input column, OneHotEncoder returns a one-hot-encoded output vector column. It is common to use Vector Assembler to combine these vectors into a single feature vector. But, since I've already converted status to a dummy variable, what's the point of the OneHotEncoder in this case?
OneHotEncoder is used to convert a Country into a numerical column. The reason could be that it isn’t currently used but will be in the future when predicting a country's life expectancy.
 
•	In the above code numerical col= all the required feature.
•	Vector assembler is being used to assemble all the features into one vector from multiple columns.
•	Next is importing the ML pipeline. A Pipeline is defined as a series of stages, each of which is either a Transformer or an Estimator. These stages are executed sequentially, and the input Data Frame is transformed as it passes through each one. On the Data Frame, the transform () method is called for the Transformer stages. For Estimator stages, the fit () method is invoked to generate a Transformer (which becomes part of the Pipeline Model or fitted Pipeline), and the transform () method of that Transformer is invoked on the Data Frame.
•	The next step is predicting life expectancy by using the Decision Tree which is discussed in the machine learning prediction section


6. Performance evaluation and comparison of methods
Performance Evaluation: - Random Forest (PySpark)
The model is trained to find out the life expectancy and factors that affect it. Next, the trained model is applied to the test dataset to predict the accuracy of the model. The model gives the following output
Root mean Square Error is 3.197
Mean squared error is 10.220
R squared is 0.884
The above result interprets the model's performance as Root Mean Square Error (RMSE) is the standard deviation of the prediction errors and it measures how far the data points are from the regression line and Mean Squared Error (MSE) shows the average square difference between predicted values and actual values.  R squared is 0.88, our model predicts a country's life expectancy with 88 percent accuracy.
Random forest With Principal component (Rmarkdown):
 
Following the Principal Component Analysis in RMarkdown, the Random Forest algorithm is used to train the model. The test set is used to predict life expectancy, the above image depicts the model's output. The mean squared error is 11.32, and the R squared value is 86.98. This means that the trained model predicts life expectancy with an 86.98 percent accuracy. It also provides the number of trees, which is 1000, as well as the number of variables tried at each step, which is 2.
Based on model performance, we can see that our model produces nearly identical results with the principal component and with the actual dataset feature. when there is only a minor difference between both outputs (R and PySpark).

Comparison of methods (With other Group members):
Student ID	ML algorithm used	Result


Decision Tree	R2 – 0.99
MSE – 0.376
RMSE – 0.613


	R2 – 1.0
MSE – 2.43
RMSE- 1.56

	
Multiple Regression	R2 - 0.60
MSE - 28.26
RMSE - 5.31

	
Multiple linear Regression	R2 - 0.62
MSE - 10.24
RMSE - 3.20


Random Forest	R2 – 0.88
MSE – 10.22
RMSE – 3.19

The table above compares the performance of various models on the same dataset. As we can see, there are three regression models, each of which yielded a different result. The data preprocessing step could be the reason. Because everyone performs different EDA steps, the results on the same dataset vary.
Two of the models are the best fit model because their R2 value is 1 and 0.99, indicating that the model was successfully implemented and accurately predicts a value. While comparing both best fit models it can be observed that the Decision Tree is performing well as compared with multiple linear regression as there is a difference in MSE and RMSE values. As per the evaluation measures, Decision tree predicted values are near the regression line as there is less difference in predicted values and actual values. whereas the random forest performed well in comparison with the rest of the two models. The rest of the two multiple regression accuracy is very low as compared with other models.  There is a very slight difference between them when compared to each other.

7. Discussion of the findings:
In our study, we used different machine algorithms to determine the best model to predict life expectancy. We discovered that the best model is the Decision Tree by comparing R2 coefficients of measures of each model. Our findings enable us to identify the best model for predicting life expectancy, as well as which factors have the greatest influence on life expectancy. According to our findings, the factors that have the greatest impact on life expectancy are Alcohol, immunization, infant deaths, HIV/AIDS, GDP, schooling, and income composition of resources.
A new insight gained from this study is data analysis reveals that your model's performance is dependent not only on the Machine Learning algorithm you used, the number of records or data you have, or how effectively you used the various techniques to analyze but also on the quality of the data. If your data is of high quality, your model will perform better in terms of accuracy and performance. It can be stated that Exploratory data analysis, Data cleaning and preprocessing plays a vital role in model implementation.
