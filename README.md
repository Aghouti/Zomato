# Zomato

📌 Project Overview

This project explores and models restaurant ratings in Bangalore using the Zomato Restaurants Dataset from Kaggle.
The goal is to analyze key factors that influence ratings and build predictive models to estimate restaurant ratings based on their features.
📊 Dataset

Size: 50,000+ restaurant entries

Features:

Cuisines

Location

Review list

Votes (numerical)

Rating (target variable)

Source: Kaggle Dataset Link (https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)

🔍 Exploratory Data Analysis

Performed a detailed EDA guided by the business questions provided in the Kaggle dataset’s PDF.
Key steps:

Handled outliers and missing values

Converted categorical features to usable formats

Created multiple visualizations to uncover insights.

![EDA Chart](TopCuisine.png)
🤖 Modeling

Preprocessed data using One-Hot Encoding for categorical features and scaling for numerical ones.

Tried multiple models:

Random Forest Classifier

Neural Network (Keras)

Evaluated using accuracy and confusion matrix.

⚠️ Challenges

Managing high-cardinality categorical features (many unique values)

Dealing with NaN and outlier values

Preventing overfitting in Neural Networks

📈 Results

Random Forest Accuracy: 96%

Neural Network Accuracy: 74%

🚀 How to Run
pip install -r requirements.txt
streamlit run app.py
