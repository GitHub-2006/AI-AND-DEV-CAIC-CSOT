from flask import Flask
import os
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-key-for-development')
from main import routes