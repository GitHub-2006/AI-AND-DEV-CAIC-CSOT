from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from textblob import TextBlob
import pandas as pd
import numpy as np
import joblib
from main import app, tweet_generator
from datetime import datetime

SimpleTweetGenerator = tweet_generator.SimpleTweetGenerator()
df = pd.read_csv("main/user_stats.csv")
values = pd.read_csv("main/inferred_company_encoded_values.csv")
values = np.array(values)
values_company = values[:, 0]
values_encoded = values[:, 1]
file = "main/like_predictor.pkl"
model = joblib.load(file)

class PredictionForm(FlaskForm):
    username = StringField('Username')
    content = StringField('Content of the post')
    datetime = StringField('Day and time of the post')
    company = StringField('Inferred company')
    media = StringField('Media files link (optional)')
    submit = SubmitField('Predict')
class TweetForm1(FlaskForm):
    company = StringField('Company Name', render_kw={"placeholder": "Enter company name"})
    tweet_type = StringField('Tweet Type', render_kw={"placeholder": "Announcement, Question, General"})
    message = StringField('Tweet Message', render_kw={"placeholder": "Enter your message"})
    topic = StringField('Topic', render_kw={"placeholder": "Enter topic for the tweet"})
    submit = SubmitField('Generate')
class TweetForm2(FlaskForm):
    username = StringField('Username', render_kw={"placeholder": "Enter your username"})
    company = StringField('Company Name', render_kw={"placeholder": "Enter company name"})
    tweet_type = StringField('Tweet Type', render_kw={"placeholder": "Announcement, Question, General"})
    message = StringField('Tweet Message', render_kw={"placeholder": "Enter your message"})
    topic = StringField('Topic', render_kw={"placeholder": "Enter topic for the tweet"})
    submit = SubmitField('Generate and Predict')

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    form = PredictionForm()
    if form.validate_on_submit():
        username = form.username.data
        post_content = form.content.data
        post_datetime = form.datetime.data
        inferred_company = form.company.data
        inferred_company_encoded = int(values_encoded[list(values_company).index(inferred_company)]) if inferred_company in values_company else -1
        has_mention = int('@' in post_content or '#' in post_content)
        word_count = len(post_content.split())
        content_length = len(post_content)
        Is_weekend = int(post_datetime.lower() in ['saturday', 'sunday'])
        Release__time_year = int(post_datetime.split('-')[0])
        sentiment = TextBlob(post_content).sentiment.polarity
        if username in df['Username'].values:
            Average_Likes_Post = float(df[df['Username'] == username]['Average_Likes_Post'].values[0])
            User_Post_Count = int(df[df['Username'] == username]['User_Post_Count'].values[0])
        else:
            Average_Likes_Post = 0
            User_Post_Count = 0
        features = {
            'Average_Likes_Post': Average_Likes_Post,
            'User_Post_Count': User_Post_Count,
            'Word_Count': word_count,
            'Inferred_Company_Encoded': inferred_company_encoded,
            'Content_Length': content_length,
            'Has_Mention': has_mention,
            'Is_Weekend': Is_weekend,
            'Release_Time_Year': Release__time_year,
            'Sentiment': sentiment
        }
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        prediction = int(np.exp(prediction)[0])
        return render_template('predict.html', form=form, prediction=prediction)
    return render_template('predict.html', form=form)

@app.route('/tweet', methods=['GET', 'POST'])
def tweet_page():
    form = TweetForm1()
    if form.validate_on_submit():
        company = form.company.data
        tweet_type = form.tweet_type.data
        message = form.message.data
        topic = form.topic.data
        simple_generated_tweet = SimpleTweetGenerator.generate_tweet(company, tweet_type, message, topic)
        return render_template('tweet.html', form=form, simple_generated_tweet=simple_generated_tweet)
    return render_template('tweet.html', form=form)

@app.route('/tweet_and_predict', methods=['GET', 'POST'])
def tweet_and_predict_page():
    form = TweetForm2()
    if form.validate_on_submit():
        username = form.username.data
        company = form.company.data
        tweet_type = form.tweet_type.data
        message = form.message.data
        topic = form.topic.data
        simple_generated_tweet = SimpleTweetGenerator.generate_tweet(company, tweet_type, message, topic)
        inferred_company_encoded = int(values_encoded[list(values_company).index(company)]) if company in values_company else -1
        if username in df['Username'].values:
            Average_Likes_Post = float(df[df['Username'] == username]['Average_Likes_Post'].values[0])
            User_Post_Count = int(df[df['Username'] == username]['User_Post_Count'].values[0])
        else:
            Average_Likes_Post = 0
            User_Post_Count = 0
        has_mention_simple = int('@' in simple_generated_tweet or '#' in simple_generated_tweet)
        word_count_simple = len(simple_generated_tweet.split())
        content_length_simple = len(simple_generated_tweet)
        sentiment_simple = TextBlob(simple_generated_tweet).sentiment.polarity
        Is_weekend = int(datetime.now().weekday() >= 5)
        Release__time_year = datetime.now().year
        features_simple = {
            'Average_Likes_Post': Average_Likes_Post,
            'User_Post_Count': User_Post_Count,
            'Word_Count': word_count_simple,
            'Inferred_Company_Encoded': inferred_company_encoded,
            'Content_Length': content_length_simple,
            'Has_Mention': has_mention_simple,
            'Is_Weekend': Is_weekend,
            'Release_Time_Year': Release__time_year,
            'Sentiment': sentiment_simple
        }
        features_df_simple = pd.DataFrame([features_simple])
        prediction_simple = model.predict(features_df_simple)
        prediction_simple = int(np.exp(prediction_simple)[0])
        return render_template('tweet_and_predict.html', form=form,
                               simple_generated_tweet=simple_generated_tweet,
                               prediction_simple=prediction_simple)
    return render_template('tweet_and_predict.html', form=form)