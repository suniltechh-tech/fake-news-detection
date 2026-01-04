Fake News Detection using Machine Learning
##Project Overview

This project is a Machine Learning–based Fake News Detection system that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques.

The model analyzes the textual content of news articles and predicts their authenticity with good accuracy. A simple and interactive Streamlit web application is provided for user interaction.

 ##Technologies Used

Python

Pandas, NumPy

Scikit-learn

TF-IDF Vectorization

Logistic Regression

Streamlit

##Dataset

The model was trained on a large real-world Fake News dataset.
Due to file size limitations, the dataset is not included in this repository.

##Machine Learning Approach

Text preprocessing and cleaning

Feature extraction using TF-IDF

Model training using Logistic Regression

Evaluation using train-test split

Model saved using pickle for reuse

##Project Structure
Fake-News-Detection/
│
├── app.py                # Streamlit web application
├── train_model.py        # Model training script
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation

##How to Run the Project
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Train the Model
python train_model.py

Step 3: Run the Application
streamlit run app.py

##How It Works

User enters a news article text

Text is converted into numerical features using TF-IDF

The trained model predicts the news type

Output is displayed as Real News or Fake News

##Model Details

Algorithm: Logistic Regression

Feature Extraction: TF-IDF

Output Classes: Real / Fake

Performance: High accuracy on standard datasets

##Future Enhancements

Use advanced models like BERT or LSTM

Add news URL analysis

Improve UI and visualization

Deploy on cloud platforms

Note: model.pkl and vectorizer.pkl are generated after running train_model.py

