# Likes Prediction Using Hugging Face API Models - AI Project

Overview: This project focuses on predicting the number of likes for models available on the Hugging Face API. The prediction is based on various features extracted from the Hugging Face API, and the model is trained using H2O's Random Forest algorithm. The objective is to build a predictive model that can estimate the popularity of Hugging Face models.

Key Features:
Data Fetching and Processing: The code fetches multiple pages of data from the Hugging Face API using pagination.
Data Analysis: Performs data analysis on the obtained dataset.
Data Flattening and Cleaning: Unnests specified columns, converts character columns to appropriate types, and saves the final processed data in a CSV file.
Tag Analysis: The code analyzes the tags associated with different models and creates a wide-format dataframe for further exploration.
Random Forest Model: Utilizes the H2O library to train a random forest model for predicting the number of likes based on various features.

Dependencies:
Several R libraries, including httr, dplyr, purrr, ggplot2, corrplot, plotly, h2o, skimr, recipes, stringr, and kableExtra.
Hugging Face Model API for fetching data
H2O library for building and training the random forest model
