from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

from main import routes