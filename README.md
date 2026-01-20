# sentiment-analysis-task-4

Company: Codtech IT Solutions Private Limited

Name: Yashwanth Koppera

Intern ID: CTIS1714

Domain: Data Analytics

Duration: 4 Weeks

Mentor: Neela Santhosh Kumar

1.Overview

 This repository contains Task-4 of the CODTECH Internship, which focuses on performing Sentiment Analysis on textual data using Natural Language Processing (NLP) and Machine Learning.
 The task was implemented using Google Colab, allowing efficient execution and visualization without local environment setup.

 The objective is to classify text reviews into positive or negative sentiments by applying preprocessing techniques, feature extraction, and supervised learning models.

 2.Objective

  Perform sentiment analysis on textual data

  Apply NLP preprocessing techniques

  Convert text into numerical features using TF-IDF

  Train a machine learning classification model

  Evaluate model performance using standard metrics

  Generate insights from sentiment predictions

3.Dataset

  IMDB Movie Reviews Dataset

  The dataset contains 50,000 movie reviews, each labeled as positive or negative, making it suitable for sentiment classification tasks.

Dataset Source (Kaggle):
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

4.Tools & Technologies

 Python

 Google Colab

 Pandas & NumPy

 NLTK (Text Preprocessing)

 Scikit-learn (Machine Learning)

 Matplotlib & Seaborn

 5.Methodology
  1. Data Loading

    The dataset was loaded in Google Colab using a robust CSV parsing method to handle malformed text entries.

  2. Text Preprocessing

    The following NLP techniques were applied:

    Conversion to lowercase

    Removal of punctuation and numbers

    Removal of stopwords

    Tokenization and text normalization

  3. Feature Extraction

    Text data was converted into numerical features using TF-IDF Vectorization, enabling effective machine learning processing.

  4. Model Training

    A Logistic Regression model was trained to classify movie reviews into positive or negative sentiments.

  5. Model Evaluation

    The trained model was evaluated using:

    Accuracy
 
    Precision

    Recall

    F1-Score

    Confusion Matrix

  6. Prediction & Insights

   The model was tested on sample reviews to demonstrate sentiment prediction capability.

  6.Output:

  
