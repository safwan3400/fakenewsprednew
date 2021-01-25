
from flask import Flask, request, jsonify, render_template


from feature import *


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    news_title = request.form['news_title']
    author_name = request.form['author_name']
    text = request.form['text']
    statement = news_title + " " + author_name + " " + text
    prd = fakenews(statement)

    return render_template('index.html', prediction_text=prd)



if __name__ == "__main__":
    app.run(debug=True)
