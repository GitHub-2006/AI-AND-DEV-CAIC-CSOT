# 🚀 Tweet Intelligence Engine: Project Report

## Overview

This project, developed as part of the CAIC Summer of Technology 2025 - ML+Dev track, is a full-stack AI application designed to generate tweets and predict their engagement (likes). It demonstrates the integration of machine learning models with a web interface, allowing users to interact with a simple tweet generation and engagement prediction in real-time. AI-generator has been omitted due to heavy consumption of space, leading to more memory needed than the 512MB limit set on render. 

## 🌟 Key Features

* **Tweet Generation:**
    * **Simple Generator:** Creates tweets based on predefined templates using company name, tweet type, message, and topic.
* **Like Prediction:**
    * Utilises a pre-trained machine learning model (`like_predictor.pkl`) to estimate the number of likes a generated tweet might receive.
    * Features used for Prediction include tweet content length, word count, sentiment polarity, and user-specific historical data.
* **Interactive Web Interface:**
    * A user-friendly Flask-based interface allows users to input parameters for tweet generation and immediately see the generated tweet and its predicted likes.
* **Robust Backend:**
    * Built with Flask, the backend handles API requests for tweet generation and like Prediction, integrating the machine learning model and tweet generation logic.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** scikit-learn, joblib, pandas, TextBlob
* **Frontend:** HTML, CSS (via Flask templates)
* **Deployment:** Render (Flask app)
* **Version Control:** Git, GitHub

## 📊 How it Works

The application follows a precise flow:

1.  **User Input:** The user provides details such as company name, tweet type, message, and topic through the web interface.
2.  **Tweet Generation:** The application either uses the `SimpleTweetGenerator` to craft a tweet.
3.  **Feature Extraction:** The generated tweet's characteristics (e.g., word count, character count, sentiment polarity) are extracted. Additional user-specific data (like `Average_Likes_Post`, `User_Post_Count` from `user_stats.csv`) and company encoding (`inferred_company_encoded_values.csv`) are retrieved.
4.  **Like Prediction:** These extracted features are fed into the pre-trained `like_predictor.pkl` machine learning model.
5.  **Results Display:** The generated tweet and its predicted number of likes are displayed to the user on the web page.

## 📂 Project Structure

WEEK 5/
├── main/
    ├──templates
    ├──__init__.py
    ├──routes.py
    ├──like_predictor.pkl
    ....

├── PROJECT_REPORT.md

├── app.py

├── like_predictor.pkl

├── requirements.txt

## 🚀 Running the Application Locally

To set up and run the Tweet Intelligence Engine on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/tweet-intelligence-engine.git](https://github.com/yourusername/tweet-intelligence-engine.git)
    cd tweet-intelligence-engine
    ```
2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    cd "AI + DEV/WEEK 5/" && pip install -r requirements.txt
    ```
    (Note: The `requirements.txt` file should contain all necessary libraries like `Flask`, `joblib`, `numpy`, `pandas`, `textblob`, `transformers`, `torch`, `Flask-WTF`.)

3.  **Run the Application:**
    The `app.py` is the entry point for the combined Flask application.
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/`.

## 🌐 Live Demo

The application is deployed online and can be accessed at:

**https://socialengagement.onrender.com**

## ✅ Deliverables Achieved

* **Deployed Application:** The complete system is deployed to the web and is accessible via a URL. It functions as expected, allowing users to generate tweets and view predicted likes without crashes.
* **Comprehensive Documentation:** A clean GitHub repository with a detailed `README.md` (and this `PROJECT_REPORT.md`) providing setup instructions, project overview, and functionality.
* **Organised and Readable Code:** The codebase is structured logically, clearly separating concerns (tweet generation, prediction logic, Flask routes).

## 🙏 Acknowledgements

This project was developed during the CAIC Summer of Technology 2025, a program that provided invaluable guidance and resources for learning and applying ML and development skills.
